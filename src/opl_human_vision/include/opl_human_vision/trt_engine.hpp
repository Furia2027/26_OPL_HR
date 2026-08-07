#ifndef OPL_HUMAN_VISION__TRT_ENGINE_HPP_
#define OPL_HUMAN_VISION__TRT_ENGINE_HPP_

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace opl_human_vision {

#ifdef HAVE_TENSORRT
class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }
};

struct TrtDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      delete obj;
    }
  }
};
#endif

struct Detection {
  int class_id;
  float confidence;
  cv::Rect box;
};

class TrtEngine {
public:
  TrtEngine(const std::string& engine_path,
            float conf_thresh = 0.45f,
            float nms_thresh = 0.50f,
            int device_id = 0);
  ~TrtEngine();

  bool loadEngine();
  std::vector<Detection> infer(const cv::Mat& frame);

private:
  void cleanup();
  void preprocess(const cv::Mat& frame);
  std::vector<Detection> postprocess(const cv::Size& original_shape);

  std::string engine_path_;
  float conf_thresh_;
  float nms_thresh_;
  int device_id_{0};

#ifdef HAVE_TENSORRT
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_{nullptr};

  cudaStream_t stream_{nullptr};
  void* gpu_input_buffer_{nullptr};
  void* gpu_output_buffer_{nullptr};

  std::string input_name_;
  std::string output_name_;

  // Dynamic dimension tracking
  int input_w_{640};
  int input_h_{640};
  int num_channels_{84};
  int num_anchors_{8400};
  bool is_pose_model_{false};

  size_t input_size_{0};
  size_t output_size_{0};

  // Safe container for CPU retrieval
  std::vector<float> cpu_output_buffer_;
#endif
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__TRT_ENGINE_HPP_
