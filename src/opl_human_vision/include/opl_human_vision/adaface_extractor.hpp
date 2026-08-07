#ifndef OPL_HUMAN_VISION__ADAFACE_EXTRACTOR_HPP_
#define OPL_HUMAN_VISION__ADAFACE_EXTRACTOR_HPP_

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#endif

namespace opl_human_vision {

class AdaFaceExtractor {
public:
  explicit AdaFaceExtractor(const std::string& model_path, int device_id = 0);
  ~AdaFaceExtractor();

  bool init();
  cv::Mat alignFace(const cv::Mat& src, const std::vector<cv::Point2f>& landmarks);
  std::vector<float> extract(const cv::Mat& aligned_face);

private:
  void normalizeL2(std::vector<float>& embedding);
  void cleanup();

  std::string model_path_;
  int device_id_{0};
  int input_w_{112};
  int input_h_{112};
  int feature_dim_{512};

#ifdef HAVE_TENSORRT
  // Private nested types scoped strictly to AdaFaceExtractor
  struct TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
      if (severity <= Severity::kWARNING) {
        std::cout << "[AdaFaceExtractor] " << msg << std::endl;
      }
    }
  };

  struct TrtDeleter {
    template <typename T>
    void operator()(T* obj) const {
      if (obj) {
#if NV_TENSORRT_MAJOR >= 10
        delete obj;
#else
        obj->destroy();
#endif
      }
    }
  };

  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_{nullptr};
  cudaStream_t stream_{nullptr};

  void* gpu_buffers_[2]{nullptr, nullptr};
  float* cpu_input_buffer_{nullptr};   // Allocated pinned host memory
  float* cpu_output_buffer_{nullptr};  // Allocated pinned host memory
#endif
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__ADAFACE_EXTRACTOR_HPP_
