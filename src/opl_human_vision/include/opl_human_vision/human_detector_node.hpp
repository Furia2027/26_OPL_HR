#ifndef OPL_HUMAN_VISION__HUMAN_DETECTOR_NODE_HPP_
#define OPL_HUMAN_VISION__HUMAN_DETECTOR_NODE_HPP_

#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// TensorRT
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include "opl_interfaces/msg/tracked_human_array.hpp"

namespace opl_human_vision {

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT YOLO-Pose] " << msg << std::endl;
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

class HumanDetectorNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit HumanDetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~HumanDetectorNode() override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State & state) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  bool loadTensorRTEngine(const std::string& model_path);
  void releaseResources();

  std::string image_topic_;
  std::string detections_topic_;
  std::string model_path_;

  float conf_thresh_{0.45f};
  float nms_thresh_{0.50f};
  int target_class_id_{0};
  int device_id_{0};

  // TensorRT Smart Pointers
  Logger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_{nullptr};
  cudaStream_t stream_{nullptr};

  void* gpu_buffers_[2]{nullptr, nullptr};
  float* cpu_input_buffer_{nullptr};  // Allocated as pinned host memory (cudaMallocHost)
  float* cpu_output_buffer_{nullptr}; // Allocated as pinned host memory (cudaMallocHost)

  // Tensor Names for TensorRT 10+ Compatibility
  std::string input_tensor_name_;
  std::string output_tensor_name_;

  // Dynamic Engine Dimensions
  int input_w_{640};
  int input_h_{640};
  int num_anchors_{8400};
  int num_channels_{56};

  rclcpp::CallbackGroup::SharedPtr image_cb_group_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<opl_interfaces::msg::TrackedHumanArray>::SharedPtr detection_pub_;
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__HUMAN_DETECTOR_NODE_HPP_
