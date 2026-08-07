#ifndef OPL_HUMAN_VISION__OSNET_EXTRACTOR_HPP_
#define OPL_HUMAN_VISION__OSNET_EXTRACTOR_HPP_

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

class OsnetExtractor {
public:
  explicit OsnetExtractor(const std::string& model_path, int device_id = 0);
  ~OsnetExtractor();

  bool init();
  std::vector<float> extract(const cv::Mat& human_crop);

private:
  void normalizeL2(std::vector<float>& embedding);
  void cleanup();

  std::string model_path_;
  int device_id_{0};
  int input_w_{128};
  int input_h_{256};
  int feature_dim_{512};

#ifdef HAVE_TENSORRT
  // Private nested types scoped strictly to OsnetExtractor
  struct TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
      if (severity <= Severity::kWARNING) {
        std::cout << "[OsnetExtractor] " << msg << std::endl;
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

#endif  // OPL_HUMAN_VISION__OSNET_EXTRACTOR_HPP_
