#include "opl_human_vision/human_detector_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <algorithm>
#include <fstream>
#include <cstring>

namespace opl_human_vision {

HumanDetectorNode::HumanDetectorNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("human_detector_node", options) {
  declare_parameter<std::string>("image_topic", "camera/image_raw");
  declare_parameter<std::string>("detections_topic", "raw_detections");
  declare_parameter<std::string>("model_path", "");
  declare_parameter<float>("confidence_threshold", 0.45f);
  declare_parameter<float>("nms_threshold", 0.50f);
  declare_parameter<int>("target_class_id", 0);
  declare_parameter<int>("device_id", 0);
}

HumanDetectorNode::~HumanDetectorNode() {
  releaseResources();
}

void HumanDetectorNode::releaseResources() {
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
}

bool HumanDetectorNode::loadTensorRTEngine(const std::string& model_path) {
  releaseResources();

  cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "Failed to set CUDA device: %d", device_id_);
    return false;
  }

  std::ifstream file(model_path, std::ios::binary);
  if (!file.good()) {
    RCLCPP_ERROR(get_logger(), "Engine file not found: %s", model_path.c_str());
    return false;
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
  if (!runtime_) return false;

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  if (!engine_) return false;

  context_.reset(engine_->createExecutionContext());
  if (!context_) return false;

  cudaStreamCreate(&stream_);

  // FIX 1: Robust Dynamic Querying of Tensor Names and Shapes by IO Mode
#if NV_TENSORRT_MAJOR >= 10
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      input_tensor_name_ = name;
      nvinfer1::Dims dims = engine_->getTensorShape(name);
      if (dims.nbDims == 4) { input_h_ = dims.d[2]; input_w_ = dims.d[3]; }
    } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
      output_tensor_name_ = name;
      nvinfer1::Dims dims = engine_->getTensorShape(name);
      if (dims.nbDims == 3) { num_channels_ = dims.d[1]; num_anchors_ = dims.d[2]; }
    }
  }
#else
  nvinfer1::Dims input_dims = engine_->getBindingDimensions(0);
  nvinfer1::Dims output_dims = engine_->getBindingDimensions(1);
  if (input_dims.nbDims == 4) { input_h_ = input_dims.d[2]; input_w_ = input_dims.d[3]; }
  if (output_dims.nbDims == 3) { num_channels_ = output_dims.d[1]; num_anchors_ = output_dims.d[2]; }
#endif

  RCLCPP_INFO(get_logger(), "Engine Tensor Shapes: Input [%dx%d], Output Channels [%d], Anchors [%d]",
              input_w_, input_h_, num_channels_, num_anchors_);

  size_t input_size = 1 * 3 * input_h_ * input_w_ * sizeof(float);
  size_t output_size = 1 * num_channels_ * num_anchors_ * sizeof(float);

  // FIX 2: CUDA allocation status checking
  cudaError_t status1 = cudaMallocHost(reinterpret_cast<void**>(&cpu_input_buffer_), input_size);
  cudaError_t status2 = cudaMallocHost(reinterpret_cast<void**>(&cpu_output_buffer_), output_size);
  cudaError_t status3 = cudaMalloc(&gpu_buffers_[0], input_size);
  cudaError_t status4 = cudaMalloc(&gpu_buffers_[1], output_size);

  if (status1 != cudaSuccess || status2 != cudaSuccess || status3 != cudaSuccess || status4 != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "Failed to allocate CUDA memory buffers!");
    releaseResources();
    return false;
  }

  return true;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanDetectorNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "Configuring HumanDetectorNode...");

  image_topic_ = get_parameter("image_topic").as_string();
  detections_topic_ = get_parameter("detections_topic").as_string();
  model_path_ = get_parameter("model_path").as_string();
  conf_thresh_ = static_cast<float>(get_parameter("confidence_threshold").as_double());
  nms_thresh_ = static_cast<float>(get_parameter("nms_threshold").as_double());
  target_class_id_ = get_parameter("target_class_id").as_int();
  device_id_ = get_parameter("device_id").as_int();

  if (model_path_.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter 'model_path' is required but not set.");
    return CallbackReturn::FAILURE;
  }

  RCLCPP_INFO(get_logger(), "Loading YOLO-Pose TensorRT model on GPU %d from: %s", device_id_, model_path_.c_str());

  if (!loadTensorRTEngine(model_path_)) {
    RCLCPP_ERROR(get_logger(), "Failed to load TensorRT Pose Engine!");
    return CallbackReturn::FAILURE;
  }

  detection_pub_ = create_publisher<opl_interfaces::msg::TrackedHumanArray>(
    detections_topic_, rclcpp::SensorDataQoS()
  );

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanDetectorNode::on_activate(const rclcpp_lifecycle::State & state) {
  LifecycleNode::on_activate(state);

  detection_pub_->on_activate();

  image_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = image_cb_group_;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    image_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&HumanDetectorNode::imageCallback, this, std::placeholders::_1),
    sub_options
  );

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanDetectorNode::on_deactivate(const rclcpp_lifecycle::State & state) {
  LifecycleNode::on_deactivate(state);

  image_sub_.reset();
  detection_pub_->on_deactivate();

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanDetectorNode::on_cleanup(const rclcpp_lifecycle::State &) {
  detection_pub_.reset();
  releaseResources();
  return CallbackReturn::SUCCESS;
}

void HumanDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  if (!detection_pub_->is_activated() || !engine_) {
    return;
  }

  try {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    if (cv_ptr->image.empty()) {
      return;
    }

    cv::Mat img = cv_ptr->image;
    int img_w = img.cols;
    int img_h = img.rows;

    // 1. Letterbox Preprocessing
    float scale = std::min(static_cast<float>(input_w_) / img_w, static_cast<float>(input_h_) / img_h);
    int unpad_w = std::round(img_w * scale);
    int unpad_h = std::round(img_h * scale);
    int pad_x = (input_w_ - unpad_w) / 2;
    int pad_y = (input_h_ - unpad_h) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(unpad_w, unpad_h));
    cv::Mat canvas(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, unpad_w, unpad_h)));

    // 2. Optimized Preprocessing: BGR->RGB conversion and HWC->CHW transposition
    cv::Mat blob = cv::dnn::blobFromImage(
        canvas, 1.0 / 255.0, cv::Size(input_w_, input_h_),
        cv::Scalar(0, 0, 0), true, false
    );
    size_t channel_size = input_w_ * input_h_;
    std::memcpy(cpu_input_buffer_, blob.ptr<float>(), 3 * channel_size * sizeof(float));

    // 3. CUDA Async Inference
    cudaSetDevice(device_id_);
    cudaMemcpyAsync(gpu_buffers_[0], cpu_input_buffer_, 3 * channel_size * sizeof(float), cudaMemcpyHostToDevice, stream_);

#if NV_TENSORRT_MAJOR >= 10
    context_->setTensorAddress(input_tensor_name_.c_str(), gpu_buffers_[0]);
    context_->setTensorAddress(output_tensor_name_.c_str(), gpu_buffers_[1]);
    context_->enqueueV3(stream_);
#else
    void* bindings[] = {gpu_buffers_[0], gpu_buffers_[1]};
    context_->enqueueV2(bindings, stream_, nullptr);
#endif

    cudaMemcpyAsync(cpu_output_buffer_, gpu_buffers_[1], num_channels_ * num_anchors_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 4. Postprocessing: Parse Predictions
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point3f>> keypoints_list;

    for (int i = 0; i < num_anchors_; ++i) {
      float conf = cpu_output_buffer_[4 * num_anchors_ + i];
      if (conf < conf_thresh_) continue;

      float cx = cpu_output_buffer_[0 * num_anchors_ + i];
      float cy = cpu_output_buffer_[1 * num_anchors_ + i];
      float w = cpu_output_buffer_[2 * num_anchors_ + i];
      float h = cpu_output_buffer_[3 * num_anchors_ + i];

      int x = static_cast<int>((cx - w / 2.0f - pad_x) / scale);
      int y = static_cast<int>((cy - h / 2.0f - pad_y) / scale);
      int width = static_cast<int>(w / scale);
      int height = static_cast<int>(h / scale);

      std::vector<cv::Point3f> kps;
      for (int k = 0; k < 17; ++k) {
        float kx = (cpu_output_buffer_[(5 + k * 3 + 0) * num_anchors_ + i] - pad_x) / scale;
        float ky = (cpu_output_buffer_[(5 + k * 3 + 1) * num_anchors_ + i] - pad_y) / scale;
        float kc = cpu_output_buffer_[(5 + k * 3 + 2) * num_anchors_ + i];
        kps.emplace_back(kx, ky, kc);
      }

      bboxes.emplace_back(x, y, width, height);
      confidences.push_back(conf);
      keypoints_list.push_back(kps);
    }

    // 5. NMS (Non-Maximum Suppression)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confidences, conf_thresh_, nms_thresh_, indices);

    opl_interfaces::msg::TrackedHumanArray detections_msg;
    detections_msg.header = msg->header;

    for (int idx : indices) {
      const auto& kps = keypoints_list[idx];

      float min_x = static_cast<float>(img_w), min_y = static_cast<float>(img_h);
      float max_x = 0.0f, max_y = 0.0f;
      int valid_kps = 0;

      for (const auto& kp : kps) {
        if (kp.z > 0.35f) {
          min_x = std::min(min_x, kp.x);
          min_y = std::min(min_y, kp.y);
          max_x = std::max(max_x, kp.x);
          max_y = std::max(max_y, kp.y);
          valid_kps++;
        }
      }

      cv::Rect final_box = bboxes[idx];

      if (valid_kps >= 4) {
        float kp_w = max_x - min_x;
        float kp_h = max_y - min_y;

        int tx1 = static_cast<int>(min_x - kp_w * 0.06f);
        int ty1 = static_cast<int>(min_y - kp_h * 0.08f);
        int tx2 = static_cast<int>(max_x + kp_w * 0.06f);
        int ty2 = static_cast<int>(max_y + kp_h * 0.06f);

        if (tx2 > tx1 && ty2 > ty1) {
          final_box = cv::Rect(tx1, ty1, tx2 - tx1, ty2 - ty1);
        }
      }

      // 6. Clamp to image boundary
      int x1 = std::max(0, std::min(final_box.x, img_w - 1));
      int y1 = std::max(0, std::min(final_box.y, img_h - 1));
      int x2 = std::max(0, std::min(final_box.x + final_box.width, img_w));
      int y2 = std::max(0, std::min(final_box.y + final_box.height, img_h));

      int clamped_w = x2 - x1;
      int clamped_h = y2 - y1;

      if (clamped_w < 10 || clamped_h < 10) continue;

      opl_interfaces::msg::TrackedHuman human_msg;
      human_msg.confidence = confidences[idx];
      human_msg.bbox.x_offset = x1;
      human_msg.bbox.y_offset = y1;
      human_msg.bbox.width = clamped_w;
      human_msg.bbox.height = clamped_h;

      detections_msg.humans.push_back(human_msg);
    }

    detection_pub_->publish(detections_msg);

  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Standard exception in imageCallback: %s", e.what());
  }
}

}  // namespace opl_human_vision

RCLCPP_COMPONENTS_REGISTER_NODE(opl_human_vision::HumanDetectorNode)
