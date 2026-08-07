#include "opl_human_vision/adaface_extractor.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

namespace opl_human_vision {

AdaFaceExtractor::AdaFaceExtractor(const std::string& model_path, int device_id)
    : model_path_(model_path), device_id_(device_id) {}

AdaFaceExtractor::~AdaFaceExtractor() {
  cleanup();
}

void AdaFaceExtractor::cleanup() {
#ifdef HAVE_TENSORRT
  if (device_id_ >= 0) {
    cudaSetDevice(device_id_);
  }
  if (stream_) {
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

bool AdaFaceExtractor::init() {
#ifdef HAVE_TENSORRT
  cleanup();

  cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess) {
    std::cerr << "[AdaFaceExtractor] Failed to set CUDA device: " << device_id_ << std::endl;
    return false;
  }

  std::ifstream file(model_path_, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[AdaFaceExtractor] Failed to open engine file: " << model_path_ << std::endl;
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
    std::cerr << "[AdaFaceExtractor] Failed to create TensorRT Runtime." << std::endl;
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(stream_data.data(), size));
  if (!engine_) {
    std::cerr << "[AdaFaceExtractor] Failed to deserialize TensorRT Engine." << std::endl;
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    std::cerr << "[AdaFaceExtractor] Failed to create Execution Context." << std::endl;
    return false;
  }

  cudaStreamCreate(&stream_);

  size_t input_bytes = 1 * 3 * input_h_ * input_w_ * sizeof(float);
  size_t output_bytes = 1 * feature_dim_ * sizeof(float);

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

cv::Mat AdaFaceExtractor::alignFace(const cv::Mat& src, const std::vector<cv::Point2f>& landmarks) {
  if (landmarks.size() != 5 || src.empty()) return src;

  // Standard InsightFace 112x112 target landmark reference positions
  static const std::vector<cv::Point2f> dst_pts = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f}
  };

  cv::Mat M = cv::estimateAffinePartial2D(landmarks, dst_pts);
  cv::Mat aligned;
  if (!M.empty()) {
    cv::warpAffine(src, aligned, M, cv::Size(input_w_, input_h_));
  } else {
    cv::resize(src, aligned, cv::Size(input_w_, input_h_));
  }
  return aligned;
}

void AdaFaceExtractor::normalizeL2(std::vector<float>& embedding) {
  float sum_sq = 0.0f;
  for (float val : embedding) {
    sum_sq += val * val;
  }
  float norm = std::sqrt(sum_sq) + 1e-6f; // Epsilon prevents division-by-zero
  for (float& val : embedding) {
    val /= norm;
  }
}

std::vector<float> AdaFaceExtractor::extract(const cv::Mat& aligned_face) {
  if (aligned_face.empty()) return {};

#ifdef HAVE_TENSORRT
  if (!engine_ || !context_) return {};

  // AdaFace scaling: Normalize pixels from [0, 255] to [-1, 1] via: (x - 127.5) / 128.0
  cv::Mat blob = cv::dnn::blobFromImage(
      aligned_face, 1.0 / 128.0, cv::Size(input_w_, input_h_),
      cv::Scalar(127.5, 127.5, 127.5), true, false
  );

  size_t input_bytes = 1 * 3 * input_h_ * input_w_ * sizeof(float);
  size_t output_bytes = feature_dim_ * sizeof(float);

  // Copy input blob to pinned host memory, then transfer to GPU VRAM
  std::memcpy(cpu_input_buffer_, blob.ptr<float>(), input_bytes);

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

  // Copy result from GPU VRAM to pinned host memory, then to std::vector
  cudaMemcpyAsync(cpu_output_buffer_, gpu_buffers_[1], output_bytes, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  std::vector<float> embedding(feature_dim_);
  std::memcpy(embedding.data(), cpu_output_buffer_, output_bytes);

  // L2 Normalization
  normalizeL2(embedding);

  return embedding;
#else
  // Fallback dummy embedding if compiled without TensorRT
  std::vector<float> dummy(feature_dim_, 0.1f);
  normalizeL2(dummy);
  return dummy;
#endif
}

}  // namespace opl_human_vision
