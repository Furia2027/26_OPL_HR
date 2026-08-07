#include "opl_human_vision/scrfd_detector.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace opl_human_vision {

#ifdef HAVE_TENSORRT
struct FaceBox {
    float x1, y1, x2, y2;
    float score;
    std::vector<cv::Point2f> landmarks;
};

static float calculate_iou(const FaceBox& a, const FaceBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter_area / (area_a + area_b - inter_area + 1e-5f);
}

static void apply_nms(std::vector<FaceBox>& proposals, float iou_threshold) {
    std::sort(proposals.begin(), proposals.end(), [](const FaceBox& a, const FaceBox& b) { return a.score > b.score; });
    std::vector<FaceBox> nms_results;
    std::vector<bool> suppressed(proposals.size(), false);

    for (size_t i = 0; i < proposals.size(); ++i) {
        if (suppressed[i]) continue;
        nms_results.push_back(proposals[i]);
        for (size_t j = i + 1; j < proposals.size(); ++j) {
            if (!suppressed[j]) {
                if (calculate_iou(proposals[i], proposals[j]) > iou_threshold) suppressed[j] = true;
            }
        }
    }
    proposals = nms_results;
}

static void decode_stride_layer(
    const float* score_ptr, const float* bbox_ptr, const float* kps_ptr,
    int stride, int input_w, int input_h, float conf_threshold, bool is_score_2d,
    std::vector<FaceBox>& proposals)
{
    const int num_anchors = 2;
    int feat_w = input_w / stride;
    int feat_h = input_h / stride;

    for (int h = 0; h < feat_h; ++h) {
        for (int w = 0; w < feat_w; ++w) {
            for (int a = 0; a < num_anchors; ++a) {
                int idx = (h * feat_w + w) * num_anchors + a;
                float score = is_score_2d ? score_ptr[idx * 2 + 1] : score_ptr[idx];

                if (score < conf_threshold) continue;

                float cx = w * stride;
                float cy = h * stride;
                FaceBox box;
                box.score = score;
                box.x1 = cx - bbox_ptr[idx * 4 + 0] * stride;
                box.y1 = cy - bbox_ptr[idx * 4 + 1] * stride;
                box.x2 = cx + bbox_ptr[idx * 4 + 2] * stride;
                box.y2 = cy + bbox_ptr[idx * 4 + 3] * stride;

                for (int k = 0; k < 5; ++k) {
                    float lx = cx + kps_ptr[idx * 10 + k * 2 + 0] * stride;
                    float ly = cy + kps_ptr[idx * 10 + k * 2 + 1] * stride;
                    box.landmarks.push_back(cv::Point2f(lx, ly));
                }
                proposals.push_back(box);
            }
        }
    }
}
#endif

ScrfdDetector::ScrfdDetector(const std::string& model_path, int input_w, int input_h, int device_id)
  : model_path_(model_path), input_w_(input_w), input_h_(input_h), device_id_(device_id) {}

ScrfdDetector::~ScrfdDetector() { cleanup(); }

void ScrfdDetector::cleanup() {
#ifdef HAVE_TENSORRT
  if (device_id_ >= 0) {
    cudaSetDevice(device_id_);
  }
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (input_buffer_) {
    cudaFree(input_buffer_);
    input_buffer_ = nullptr;
  }
  if (cpu_input_buffer_) {
    cudaFreeHost(cpu_input_buffer_);
    cpu_input_buffer_ = nullptr;
  }
  for (auto& ot : output_tensors_) {
    if (ot.buffer) {
      cudaFree(ot.buffer);
      ot.buffer = nullptr;
    }
    if (ot.cpu_buffer) {
      cudaFreeHost(ot.cpu_buffer);
      ot.cpu_buffer = nullptr;
    }
  }
  output_tensors_.clear();

  context_.reset();
  engine_.reset();
  runtime_.reset();
#endif
}

bool ScrfdDetector::init() {
#ifdef HAVE_TENSORRT
  cleanup();

  if (cudaSetDevice(device_id_) != cudaSuccess) {
    std::cerr << "[ScrfdDetector] Failed to set CUDA device: " << device_id_ << std::endl;
    return false;
  }

  std::ifstream file(model_path_, std::ios::binary);
  if (!file.good()) return false;

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) { cleanup(); return false; }

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  if (!engine_) { cleanup(); return false; }

  context_.reset(engine_->createExecutionContext());
  if (!context_) { cleanup(); return false; }

  if (cudaStreamCreate(&stream_) != cudaSuccess) { cleanup(); return false; }

  output_tensors_.clear();

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* t_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(t_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = t_name;
      context_->setInputShape(input_name_.c_str(), nvinfer1::Dims4{1, 3, input_h_, input_w_});
      size_t input_bytes = 1 * 3 * input_h_ * input_w_ * sizeof(float);

      if (cudaMalloc(&input_buffer_, input_bytes) != cudaSuccess) { cleanup(); return false; }
      if (cudaMallocHost(reinterpret_cast<void**>(&cpu_input_buffer_), input_bytes) != cudaSuccess) { cleanup(); return false; }

      context_->setTensorAddress(input_name_.c_str(), input_buffer_);
    }
    else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      nvinfer1::Dims out_dims = context_->getTensorShape(t_name);
      size_t t_size = 1;
      std::vector<int64_t> dims;
      for (int d = 0; d < out_dims.nbDims; ++d) {
        int64_t dim_val = std::max<int64_t>(1, out_dims.d[d]);
        t_size *= dim_val;
        dims.push_back(dim_val);
      }
      size_t t_bytes = t_size * sizeof(float);

      void* t_buf = nullptr;
      float* cpu_t_buf = nullptr;
      if (cudaMalloc(&t_buf, t_bytes) != cudaSuccess) { cleanup(); return false; }
      if (cudaMallocHost(reinterpret_cast<void**>(&cpu_t_buf), t_bytes) != cudaSuccess) { cleanup(); return false; }

      context_->setTensorAddress(t_name, t_buf);

      OutputTensorInfo ot;
      ot.name = t_name;
      ot.buffer = t_buf;
      ot.cpu_buffer = cpu_t_buf;
      ot.size = t_size;
      ot.bytes = t_bytes;
      ot.dims = dims;
      output_tensors_.push_back(ot);
    }
  }

  if (input_name_.empty() || output_tensors_.empty()) { cleanup(); return false; }
  return true;
#else
  return true;
#endif
}

bool ScrfdDetector::detect(const cv::Mat& human_crop, std::vector<cv::Point2f>& out_landmarks, float& out_score) {
#ifdef HAVE_TENSORRT
  if (human_crop.empty() || !input_buffer_ || output_tensors_.empty()) return false;

  cudaSetDevice(device_id_);

  int search_h = std::min(human_crop.rows, static_cast<int>(human_crop.rows * 0.85));
  if (human_crop.cols < 15 || search_h < 15) return false;

  cv::Rect safe_roi(0, 0, human_crop.cols, search_h);
  cv::Mat face_roi = human_crop(safe_roi);
  cv::Mat rgb;
  cv::cvtColor(face_roi, rgb, cv::COLOR_BGR2RGB);

  float scale = std::min(static_cast<float>(input_w_) / rgb.cols, static_cast<float>(input_h_) / rgb.rows);
  int new_w = std::round(rgb.cols * scale);
  int new_h = std::round(rgb.rows * scale);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(new_w, new_h));
  cv::Mat padded = cv::Mat::zeros(input_h_, input_w_, CV_8UC3);
  resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

  cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 128.0, cv::Size(input_w_, input_h_), cv::Scalar(127.5, 127.5, 127.5), false, false);

  size_t input_bytes = blob.total() * sizeof(float);
  std::memcpy(cpu_input_buffer_, blob.ptr<float>(), input_bytes);

  cudaMemcpyAsync(input_buffer_, cpu_input_buffer_, input_bytes, cudaMemcpyHostToDevice, stream_);

#if NV_TENSORRT_MAJOR >= 10
  context_->enqueueV3(stream_);
#else
  context_->enqueueV3(stream_);
#endif

  for (size_t i = 0; i < output_tensors_.size(); ++i) {
      cudaMemcpyAsync(output_tensors_[i].cpu_buffer, output_tensors_[i].buffer, output_tensors_[i].bytes, cudaMemcpyDeviceToHost, stream_);
  }
  cudaStreamSynchronize(stream_);

  // Robust Multi-Pass Tensor Matcher
  std::vector<bool> consumed(output_tensors_.size(), false);

  auto get_tensor_data = [&](int stride, int expected_channels, const std::string& hint, bool& is_2d) -> const float* {
      int grid_w = input_w_ / stride;
      int anchors = grid_w * grid_w * 2;
      size_t target_size = anchors * expected_channels;
      size_t target_size_2d = anchors * 2; // Some models export 1-channel scores as 2-channels [bg, fg]

      // Pass 1: Try Name hint
      for (size_t i = 0; i < output_tensors_.size(); ++i) {
          if (consumed[i]) continue;
          if (output_tensors_[i].name.find(hint) != std::string::npos) {
              if (output_tensors_[i].size == target_size) { consumed[i] = true; is_2d = false; return output_tensors_[i].cpu_buffer; }
              if (expected_channels == 1 && output_tensors_[i].size == target_size_2d) { consumed[i] = true; is_2d = true; return output_tensors_[i].cpu_buffer; }
          }
      }

      // Pass 2: Shape/Grid Matching
      for (size_t i = 0; i < output_tensors_.size(); ++i) {
          if (consumed[i]) continue;
          bool size_match = (output_tensors_[i].size == target_size);
          bool size_match_2d = (expected_channels == 1 && output_tensors_[i].size == target_size_2d);

          if (size_match || size_match_2d) {
              bool shape_matches_stride = false;
              for (int64_t dim : output_tensors_[i].dims) {
                  if (dim == grid_w || dim == (grid_w * grid_w)) shape_matches_stride = true;
              }
              if (shape_matches_stride) {
                  consumed[i] = true;
                  is_2d = size_match_2d;
                  return output_tensors_[i].cpu_buffer;
              }
          }
      }

      // Pass 3: Blind Size Fallback
      for (size_t i = 0; i < output_tensors_.size(); ++i) {
          if (consumed[i]) continue;
          if (output_tensors_[i].size == target_size) { consumed[i] = true; is_2d = false; return output_tensors_[i].cpu_buffer; }
          if (expected_channels == 1 && output_tensors_[i].size == target_size_2d) { consumed[i] = true; is_2d = true; return output_tensors_[i].cpu_buffer; }
      }
      return nullptr;
  };

  std::vector<FaceBox> all_proposals;
  std::vector<int> strides = {8, 16, 32};

  for (int stride : strides) {
      bool is_score_2d = false, dummy_2d = false;
      const float* score_ptr = get_tensor_data(stride, 1, "score", is_score_2d);
      const float* bbox_ptr  = get_tensor_data(stride, 4, "bbox", dummy_2d);
      const float* kps_ptr   = get_tensor_data(stride, 10, "kps", dummy_2d);

      if (!score_ptr || !bbox_ptr || !kps_ptr) {
          std::cerr << "[ScrfdDetector] Error: Missing output tensor for stride " << stride << std::endl;
          return false;
      }
      decode_stride_layer(score_ptr, bbox_ptr, kps_ptr, stride, input_w_, input_h_, conf_threshold_, is_score_2d, all_proposals);
  }

  apply_nms(all_proposals, nms_threshold_);

  if (all_proposals.empty()) return false;

  FaceBox best_face = all_proposals[0];
  out_score = best_face.score;
  out_landmarks.clear();

  for (int k = 0; k < 5; ++k) {
      out_landmarks.push_back(cv::Point2f(best_face.landmarks[k].x / scale, best_face.landmarks[k].y / scale));
  }

  return true;
#else
  return false;
#endif
}

} // namespace opl_human_vision
