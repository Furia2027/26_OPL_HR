#ifndef OPL_HUMAN_VISION__SCRFD_DETECTOR_HPP_
#define OPL_HUMAN_VISION__SCRFD_DETECTOR_HPP_

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

struct OutputTensorInfo {
  std::string name;
  void* buffer = nullptr;        // Device (GPU VRAM) pointer
  float* cpu_buffer = nullptr;   // Pinned Host (CPU RAM) pointer
  size_t size = 0;
  size_t bytes = 0;
  std::vector<int64_t> dims;     // Physical shape dimensions
};

class ScrfdDetector {
public:
  ScrfdDetector(const std::string& model_path, int input_w = 640, int input_h = 640, int device_id = 0);
  ~ScrfdDetector();

  bool init();
  bool detect(const cv::Mat& human_crop, std::vector<cv::Point2f>& out_landmarks, float& out_score);

  void setConfThreshold(float conf_threshold) { conf_threshold_ = conf_threshold; }
  void setNmsThreshold(float nms_threshold) { nms_threshold_ = nms_threshold; }

private:
  void cleanup();

  std::string model_path_;
  int input_w_;
  int input_h_;
  int device_id_{0};

  float conf_threshold_{0.50f};
  float nms_threshold_{0.40f};

#ifdef HAVE_TENSORRT
  struct TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
      if (severity <= Severity::kERROR) {
        std::cerr << "[Scrfd TRT Error] " << msg << std::endl;
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

  void* input_buffer_{nullptr};         // Device input memory
  float* cpu_input_buffer_{nullptr};    // Pinned host input memory
  std::string input_name_;

  std::vector<OutputTensorInfo> output_tensors_;
#endif
};

} // namespace opl_human_vision

#endif // OPL_HUMAN_VISION__SCRFD_DETECTOR_HPP_
