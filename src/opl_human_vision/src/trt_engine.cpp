#include "opl_human_vision/trt_engine.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

#ifndef HAVE_TENSORRT
#include <opencv2/objdetect.hpp>
#endif

namespace opl_human_vision {

TrtEngine::TrtEngine(const std::string& engine_path, float conf_thresh, float nms_thresh, int device_id)
: engine_path_(engine_path), conf_thresh_(conf_thresh), nms_thresh_(nms_thresh), device_id_(device_id) {
  std::cout << "[TrtEngine] Initializing engine with path: " << engine_path_ << " on GPU device " << device_id_ << std::endl;
  if (!loadEngine()) {
    std::cerr << "[TrtEngine] FAILED to load engine!" << std::endl;
  }
}

TrtEngine::~TrtEngine() {
  cleanup();
  std::cout << "[TrtEngine] Resources released." << std::endl;
}

void TrtEngine::cleanup() {
#ifdef HAVE_TENSORRT
  if (device_id_ >= 0) {
    cudaSetDevice(device_id_);
  }
  if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
  if (gpu_input_buffer_) { cudaFree(gpu_input_buffer_); gpu_input_buffer_ = nullptr; }
  if (gpu_output_buffer_) { cudaFree(gpu_output_buffer_); gpu_output_buffer_ = nullptr; }

  // Unique pointers safely reset execution context and engine objects
  context_.reset();
  engine_.reset();
  runtime_.reset();
#endif
}

bool TrtEngine::loadEngine() {
#ifdef HAVE_TENSORRT
  cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess) {
    std::cerr << "[TrtEngine] Failed to set CUDA device " << device_id_ << std::endl;
    return false;
  }
  std::cout << "[TrtEngine] Loading TensorRT engine into VRAM from " << engine_path_ << std::endl;

  std::ifstream file(engine_path_, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[TrtEngine] Error reading engine file at: " << engine_path_ << std::endl;
    return false;
  }

  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> trtModelStream(size);
  file.read(trtModelStream.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) { cleanup(); return false; }

  engine_.reset(runtime_->deserializeCudaEngine(trtModelStream.data(), size));
  if (!engine_) { cleanup(); return false; }

  context_.reset(engine_->createExecutionContext());
  if (!context_) { cleanup(); return false; }

  cudaStreamCreate(&stream_);

  // Dynamically extract tensor names and dimensions
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* t_name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(t_name) == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = t_name;
      nvinfer1::Dims dims = engine_->getTensorShape(t_name);
      input_h_ = dims.d[2];
      input_w_ = dims.d[3];
      input_size_ = 1 * 3 * input_h_ * input_w_ * sizeof(float);
    } else if (engine_->getTensorIOMode(t_name) == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = t_name;
      nvinfer1::Dims dims = engine_->getTensorShape(t_name);
      num_channels_ = dims.d[1]; // 84 (COCO) or 56 (Pose)
      num_anchors_ = dims.d[2];  // Usually 8400
      output_size_ = 1 * num_channels_ * num_anchors_ * sizeof(float);
    }
  }

  if (input_name_.empty() || output_name_.empty()) { cleanup(); return false; }

  // Detect model architecture type
  is_pose_model_ = (num_channels_ == 56);
  std::cout << "[TrtEngine] Tensor Dimensions: Channels=" << num_channels_
            << ", Anchors=" << num_anchors_
            << " (" << (is_pose_model_ ? "YOLOv8-Pose" : "Standard YOLOv8") << ")" << std::endl;

  cudaMalloc(&gpu_input_buffer_, input_size_);
  cudaMalloc(&gpu_output_buffer_, output_size_);
  cpu_output_buffer_.resize(num_channels_ * num_anchors_);

  context_->setTensorAddress(input_name_.c_str(), gpu_input_buffer_);
  context_->setTensorAddress(output_name_.c_str(), gpu_output_buffer_);

  return true;
#else
  std::cout << "[TrtEngine] Running CPU Mode: Using OpenCV HOG People Detector." << std::endl;
  return true;
#endif
}

#ifdef HAVE_TENSORRT
void TrtEngine::preprocess(const cv::Mat& frame) {
  cv::Mat blob = cv::dnn::blobFromImage(
      frame, 1.0 / 255.0, cv::Size(input_w_, input_h_),
      cv::Scalar(0, 0, 0), true, false
  );

  cudaMemcpyAsync(gpu_input_buffer_, blob.ptr<float>(), input_size_, cudaMemcpyHostToDevice, stream_);
}

std::vector<Detection> TrtEngine::postprocess(const cv::Size& original_shape) {
  std::vector<Detection> detections;
  std::vector<cv::Rect> boxes;
  std::vector<float> confs;

  float x_factor = static_cast<float>(original_shape.width) / input_w_;
  float y_factor = static_cast<float>(original_shape.height) / input_h_;

  // COCO format: Person is class 0 -> Channel index 4 in both YOLOv8 standard (4..83) and Pose (4)
  const int person_class_offset = 4;

  for (int i = 0; i < num_anchors_; ++i) {
    float person_conf = cpu_output_buffer_[person_class_offset * num_anchors_ + i];

    if (person_conf > conf_thresh_) {
      float cx = cpu_output_buffer_[0 * num_anchors_ + i];
      float cy = cpu_output_buffer_[1 * num_anchors_ + i];
      float w  = cpu_output_buffer_[2 * num_anchors_ + i];
      float h  = cpu_output_buffer_[3 * num_anchors_ + i];

      int left   = static_cast<int>((cx - 0.5f * w) * x_factor);
      int top    = static_cast<int>((cy - 0.5f * h) * y_factor);
      int right  = static_cast<int>((cx + 0.5f * w) * x_factor);
      int bottom = static_cast<int>((cy + 0.5f * h) * y_factor);

      left   = std::max(0, std::min(left, original_shape.width - 1));
      top    = std::max(0, std::min(top, original_shape.height - 1));
      right  = std::max(0, std::min(right, original_shape.width - 1));
      bottom = std::max(0, std::min(bottom, original_shape.height - 1));

      int width  = right - left;
      int height = bottom - top;

      if (width > 0 && height > 0) {
        boxes.emplace_back(cv::Rect(left, top, width, height));
        confs.push_back(person_conf);
      }
    }
  }

  std::vector<int> indices;
  if (!boxes.empty()) {
    cv::dnn::NMSBoxes(boxes, confs, conf_thresh_, nms_thresh_, indices);
  }

  for (int idx : indices) {
    Detection det;
    det.class_id = 0;
    det.confidence = confs[idx];
    det.box = boxes[idx];
    detections.push_back(det);
  }

  return detections;
}
#endif

std::vector<Detection> TrtEngine::infer(const cv::Mat& frame) {
  if (frame.empty()) return {};

#ifdef HAVE_TENSORRT
  cudaSetDevice(device_id_);

  preprocess(frame);
  context_->enqueueV3(stream_);
  cudaMemcpyAsync(cpu_output_buffer_.data(), gpu_output_buffer_, output_size_, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  return postprocess(cv::Size(frame.cols, frame.rows));
#else
  std::vector<Detection> detections;
  static cv::HOGDescriptor hog;
  static bool hog_initialized = false;
  if (!hog_initialized) {
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    hog_initialized = true;
  }

  cv::Mat small_frame;
  float scale = 1.0f;
  if (frame.cols > 640) {
    scale = static_cast<float>(frame.cols) / 640.0f;
    cv::resize(frame, small_frame, cv::Size(640, static_cast<int>(frame.rows / scale)));
  } else {
    small_frame = frame;
  }

  std::vector<cv::Rect> found_boxes;
  std::vector<double> weights;
  hog.detectMultiScale(small_frame, found_boxes, weights, 0, cv::Size(8, 8), cv::Size(16, 16), 1.05, 2.0);

  for (size_t i = 0; i < found_boxes.size(); ++i) {
    if (weights[i] < 0.2) continue;
    Detection det;
    det.class_id = 0;
    det.confidence = static_cast<float>(weights[i]);
    det.box = cv::Rect(
      static_cast<int>(found_boxes[i].x * scale),
      static_cast<int>(found_boxes[i].y * scale),
      static_cast<int>(found_boxes[i].width * scale),
      static_cast<int>(found_boxes[i].height * scale)
    );
    detections.push_back(det);
  }
  return detections;
#endif
}

}  // namespace opl_human_vision
