#include "opl_human_vision/osnet_extractor.hpp"
#include "opl_human_vision/scoped_timing.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

namespace opl_human_vision {

OsnetExtractor::OsnetExtractor(const std::string& model_path, int device_id)
    : model_path_(model_path), device_id_(device_id) {}

OsnetExtractor::~OsnetExtractor() {
  cleanup();
}

void OsnetExtractor::cleanup() {
#ifdef HAVE_TENSORRT
  if (device_id_ >= 0) {
    cudaSetDevice(device_id_);
  }
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (gpu_buffers_[0]) {
    cudaFree(gpu_buffers_[0]);
    gpu_buffers_[0] = nullptr;
  }
  if (gpu_buffers_[1]) {
    cudaFree(gpu_buffers_[1]);
    gpu_buffers_[1] = nullptr;
  }
  if (cpu_input_buffer_) {
    cudaFreeHost(cpu_input_buffer_);
    cpu_input_buffer_ = nullptr;
  }
  if (cpu_output_buffer_) {
    cudaFreeHost(cpu_output_buffer_);
    cpu_output_buffer_ = nullptr;
  }

  context_.reset();
  engine_.reset();
  runtime_.reset();
#endif
}

bool OsnetExtractor::init() {
  ScopedTiming timing("OSNet.load (attempt)");
#ifdef HAVE_TENSORRT
  cleanup();

  cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess) {
    std::cerr << "[OsnetExtractor] Failed to set CUDA device: " << device_id_ << std::endl;
    return false;
  }

  std::ifstream file(model_path_, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[OsnetExtractor] Failed to open engine file: " << model_path_ << std::endl;
    return false;
  }

  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> stream_data(size);
  file.read(stream_data.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    std::cerr << "[OsnetExtractor] Failed to create TensorRT Runtime." << std::endl;
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(stream_data.data(), size));
  if (!engine_) {
    std::cerr << "[OsnetExtractor] Failed to deserialize TensorRT Engine." << std::endl;
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    std::cerr << "[OsnetExtractor] Failed to create Execution Context." << std::endl;
    return false;
  }

  cudaStreamCreate(&stream_);

  size_t input_bytes = 1 * 3 * input_h_ * input_w_ * sizeof(float);
  size_t output_bytes = feature_dim_ * sizeof(float);

  // Allocate pinned host memory
  cudaMallocHost(reinterpret_cast<void**>(&cpu_input_buffer_), input_bytes);
  cudaMallocHost(reinterpret_cast<void**>(&cpu_output_buffer_), output_bytes);

  // Allocate GPU device memory
  cudaMalloc(&gpu_buffers_[0], input_bytes);
  cudaMalloc(&gpu_buffers_[1], output_bytes);

  return true;
#else
  return true;
#endif
}

void OsnetExtractor::normalizeL2(std::vector<float>& embedding) {
  float sum_sq = 0.0f;
  for (float val : embedding) {
    sum_sq += val * val;
  }
  float norm = std::sqrt(sum_sq) + 1e-6f;
  for (float& val : embedding) {
    val /= norm;
  }
}

std::vector<float> OsnetExtractor::extract(const cv::Mat& human_crop) {
  ScopedTiming timing("OSNet.extract");
  if (human_crop.empty() || human_crop.cols < 16 || human_crop.rows < 16) {
    return {};
  }

#ifdef HAVE_TENSORRT
  if (!engine_ || !context_) return {};

  cv::Mat blob = cv::dnn::blobFromImage(
      human_crop, 1.0 / 255.0, cv::Size(input_w_, input_h_),
      cv::Scalar(0, 0, 0), true, false
  );

  // ImageNet Mean and Standard Deviation Normalization
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float std[3]  = {0.229f, 0.224f, 0.225f};

  float* blob_data = blob.ptr<float>();
  int plane_stride = input_w_ * input_h_;

  for (int c = 0; c < 3; ++c) {
    float* plane = blob_data + c * plane_stride;
    for (int i = 0; i < plane_stride; ++i) {
      plane[i] = (plane[i] - mean[c]) / std[c];
    }
  }

  size_t input_bytes = 1 * 3 * input_h_ * input_w_ * sizeof(float);
  size_t output_bytes = feature_dim_ * sizeof(float);

  // Copy processed image to pinned host memory, then transfer to GPU VRAM
  std::memcpy(cpu_input_buffer_, blob_data, input_bytes);

  cudaSetDevice(device_id_);
  cudaMemcpyAsync(gpu_buffers_[0], cpu_input_buffer_, input_bytes, cudaMemcpyHostToDevice, stream_);

#if NV_TENSORRT_MAJOR >= 10
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* tensor_name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
      context_->setTensorAddress(tensor_name, gpu_buffers_[0]);
    } else {
      context_->setTensorAddress(tensor_name, gpu_buffers_[1]);
    }
  }
  context_->enqueueV3(stream_);
#else
  void* bindings[] = {gpu_buffers_[0], gpu_buffers_[1]};
  context_->enqueueV2(bindings, stream_, nullptr);
#endif

  // Copy output from GPU VRAM to pinned host memory, then to std::vector
  cudaMemcpyAsync(cpu_output_buffer_, gpu_buffers_[1], output_bytes, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  std::vector<float> embedding(feature_dim_);
  std::memcpy(embedding.data(), cpu_output_buffer_, output_bytes);

  normalizeL2(embedding);
  return embedding;
#else
  std::vector<float> dummy(feature_dim_, 0.1f);
  normalizeL2(dummy);
  return dummy;
#endif
}

}  // namespace opl_human_vision
