#include "opl_human_vision/human_tracker_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <algorithm>
#include <vector>
#include <mutex>
#include <cmath>

namespace opl_human_vision {

HumanTrackerNode::HumanTrackerNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("human_tracker_node", options) {
  declare_parameter<std::string>("image_topic", "camera/image_raw");
  declare_parameter<std::string>("depth_topic", "camera/camera/aligned_depth_to_color/image_raw");
  declare_parameter<std::string>("camera_info_topic", "");
  declare_parameter<std::string>("detections_topic", "raw_detections");
  declare_parameter<std::string>("tracked_topic", "tracked_humans");
  declare_parameter<int>("max_age", 30);
  declare_parameter<int>("min_hits", 3);
  declare_parameter<float>("iou_threshold", 0.3f);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanTrackerNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "Configuring HumanTrackerNode...");

  image_topic_ = get_parameter("image_topic").as_string();
  depth_topic_ = get_parameter("depth_topic").as_string();
  detections_topic_ = get_parameter("detections_topic").as_string();
  tracked_topic_ = get_parameter("tracked_topic").as_string();

  int max_age = get_parameter("max_age").as_int();
  int min_hits = get_parameter("min_hits").as_int();
  float iou_threshold = static_cast<float>(get_parameter("iou_threshold").as_double());

  tracker_ = std::make_shared<TrackerEngine>(max_age, min_hits, iou_threshold);

  tracked_pub_ = create_publisher<opl_interfaces::msg::TrackedHumanArray>(tracked_topic_, rclcpp::SensorDataQoS());

  RCLCPP_INFO(get_logger(), "HumanTrackerNode Configured Successfully.");
  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanTrackerNode::on_activate(const rclcpp_lifecycle::State & state) {
  LifecycleNode::on_activate(state);
  RCLCPP_INFO(get_logger(), "Activating HumanTrackerNode...");

  tracked_pub_->on_activate();

  image_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  depth_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  track_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions img_sub_options;
  img_sub_options.callback_group = image_cb_group_;
  img_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    image_topic_, rclcpp::SensorDataQoS(),
    std::bind(&HumanTrackerNode::imageCallback, this, std::placeholders::_1),
    img_sub_options
  );

  std::string info_topic = get_parameter("camera_info_topic").as_string();
  if (info_topic.empty()) {
    info_topic = depth_topic_;
    size_t pos = info_topic.rfind("image_raw");
    if (pos != std::string::npos) {
      info_topic.replace(pos, 9, "camera_info");
    } else {
      info_topic += "/camera_info";
    }
  }

  RCLCPP_INFO(get_logger(), "Listening for Camera Info on: %s", info_topic.c_str());

  rclcpp::SubscriptionOptions info_sub_options;
  info_sub_options.callback_group = image_cb_group_;

  info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    info_topic,
    rclcpp::SensorDataQoS(),
    std::bind(&HumanTrackerNode::cameraInfoCallback, this, std::placeholders::_1),
    info_sub_options
  );

  rclcpp::SubscriptionOptions depth_sub_options;
  depth_sub_options.callback_group = depth_cb_group_;
  depth_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
    depth_topic_, rclcpp::SensorDataQoS(),
    std::bind(&HumanTrackerNode::depthCallback, this, std::placeholders::_1),
    depth_sub_options
  );

  rclcpp::SubscriptionOptions track_sub_options;
  track_sub_options.callback_group = track_cb_group_;
  track_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  detection_sub_ = create_subscription<opl_interfaces::msg::TrackedHumanArray>(
    detections_topic_, rclcpp::SensorDataQoS(),
    std::bind(&HumanTrackerNode::detectionsCallback, this, std::placeholders::_1),
    track_sub_options
  );

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanTrackerNode::on_deactivate(const rclcpp_lifecycle::State & state) {
  LifecycleNode::on_deactivate(state);
  image_sub_.reset();
  depth_sub_.reset();
  info_sub_.reset();
  detection_sub_.reset();
  tracked_pub_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanTrackerNode::on_cleanup(const rclcpp_lifecycle::State &) {
  tracked_pub_.reset();
  tracker_.reset();

  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_ptr_.reset();
  }
  {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    latest_depth_ptr_.reset();
  }

  // FIX 2: Complete parameters reset inside class member mutex lock
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    has_camera_info_ = false;
    fx_ = 0.0f;
    fy_ = 0.0f;
    cx_ = 0.0f;
    cy_ = 0.0f;
  }

  return CallbackReturn::SUCCESS;
}

void HumanTrackerNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  try {
    auto cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_ptr_ = cv_ptr;
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception on RGB: %s", e.what());
  }
}

void HumanTrackerNode::depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  try {
    auto cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
    std::lock_guard<std::mutex> lock(depth_mutex_);
    latest_depth_ptr_ = cv_ptr;
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception on Depth: %s", e.what());
  }
}

void HumanTrackerNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
  // FIX 1: Uses object member mutex instead of static global mutex
  if (!has_camera_info_) {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    if (!has_camera_info_) {
      fx_ = msg->k[0];
      cx_ = msg->k[2];
      fy_ = msg->k[4];
      cy_ = msg->k[5];
      has_camera_info_ = true;
      RCLCPP_INFO(get_logger(), "Camera Intrinsics Loaded: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", fx_, fy_, cx_, cy_);
    }
  }
}

geometry_msgs::msg::Point HumanTrackerNode::compute3DPosition(const cv::Mat& depth_img, const cv::Rect& bbox) {
  geometry_msgs::msg::Point pt;
  pt.x = 0.0; pt.y = 0.0; pt.z = 0.0;

  if (depth_img.empty() || bbox.width <= 0 || bbox.height <= 0) return pt;

  int x1 = std::max(0, std::min(bbox.x, depth_img.cols - 1));
  int y1 = std::max(0, std::min(bbox.y, depth_img.rows - 1));
  int x2 = std::max(0, std::min(bbox.x + bbox.width, depth_img.cols));
  int y2 = std::max(0, std::min(bbox.y + bbox.height, depth_img.rows));

  int safe_w = x2 - x1;
  int safe_h = y2 - y1;
  if (safe_w <= 0 || safe_h <= 0) return pt;

  int roi_w = std::max(1, static_cast<int>(safe_w * 0.40));
  int roi_h = std::max(1, static_cast<int>(safe_h * 0.40));
  int roi_x = x1 + (safe_w - roi_w) / 2;
  int roi_y = y1 + static_cast<int>(safe_h * 0.20);

  // FIX 3: Safe bounding box bounds verification
  roi_w = std::min(roi_w, depth_img.cols);
  roi_h = std::min(roi_h, depth_img.rows);
  roi_x = std::max(0, std::min(roi_x, depth_img.cols - roi_w));
  roi_y = std::max(0, std::min(roi_y, depth_img.rows - roi_h));

  cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
  cv::Mat depth_roi = depth_img(roi);

  std::vector<float> valid_depths;
  valid_depths.reserve(roi_w * roi_h);

  if (depth_img.type() == CV_16UC1) {
    for (int r = 0; r < depth_roi.rows; ++r) {
      const uint16_t* ptr = depth_roi.ptr<uint16_t>(r);
      for (int c = 0; c < depth_roi.cols; ++c) {
        if (ptr[c] > 200 && ptr[c] < 10000) {
          valid_depths.push_back(static_cast<float>(ptr[c]) * 0.001f);
        }
      }
    }
  } else if (depth_img.type() == CV_32FC1) {
    for (int r = 0; r < depth_roi.rows; ++r) {
      const float* ptr = depth_roi.ptr<float>(r);
      for (int c = 0; c < depth_roi.cols; ++c) {
        if (ptr[c] > 0.2f && ptr[c] < 10.0f && !std::isnan(ptr[c])) {
          valid_depths.push_back(ptr[c]);
        }
      }
    }
  }

  if (valid_depths.empty()) return pt;

  std::size_t n = valid_depths.size() / 2;
  std::nth_element(valid_depths.begin(), valid_depths.begin() + n, valid_depths.end());
  float median_z = valid_depths[n];

  float fx, fy, cx, cy;
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    fx = (fx_ > 0.0f) ? fx_ : depth_img.cols * 0.9375f;
    fy = (fy_ > 0.0f) ? fy_ : depth_img.rows * 1.2500f;
    cx = (cx_ > 0.0f) ? cx_ : depth_img.cols / 2.0f;
    cy = (cy_ > 0.0f) ? cy_ : depth_img.rows / 2.0f;
  }

  float center_x = x1 + safe_w / 2.0f;
  float center_y = y1 + safe_h / 2.0f;

  pt.z = median_z;
  pt.x = (center_x - cx) * median_z / fx;
  pt.y = (center_y - cy) * median_z / fy;

  return pt;
}

void HumanTrackerNode::detectionsCallback(const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg) {
  if (!tracked_pub_->is_activated()) return;

  cv_bridge::CvImageConstPtr frame_ptr_copy;
  cv_bridge::CvImageConstPtr depth_ptr_copy;

  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!latest_frame_ptr_ || latest_frame_ptr_->image.empty()) return;
    frame_ptr_copy = latest_frame_ptr_;
  }

  {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    if (latest_depth_ptr_ && !latest_depth_ptr_->image.empty()) {
      depth_ptr_copy = latest_depth_ptr_;
    }
  }

  cv::Mat frame_ref = frame_ptr_copy->image;
  cv::Mat depth_ref;

  if (depth_ptr_copy) {
    double depth_time = rclcpp::Time(depth_ptr_copy->header.stamp).seconds();
    double msg_time = rclcpp::Time(msg->header.stamp).seconds();
    if (std::abs(msg_time - depth_time) < 0.15) {
      depth_ref = depth_ptr_copy->image;
    }
  }

  std::vector<Track> active_tracks = tracker_->update(msg->humans, frame_ref);

  opl_interfaces::msg::TrackedHumanArray tracked_array_msg;
  tracked_array_msg.header = msg->header;

  for (const auto& track : active_tracks) {
    opl_interfaces::msg::TrackedHuman human;
    human.track_id = track.id;
    human.name = track.name;
    human.confidence = track.confidence;

    human.bbox.x_offset = std::max(0, static_cast<int>(track.bbox.x));
    human.bbox.y_offset = std::max(0, static_cast<int>(track.bbox.y));
    human.bbox.width = std::max(0, static_cast<int>(track.bbox.width));
    human.bbox.height = std::max(0, static_cast<int>(track.bbox.height));

    if (!depth_ref.empty()) {
      cv::Rect rect(human.bbox.x_offset, human.bbox.y_offset, human.bbox.width, human.bbox.height);
      human.position_3d = compute3DPosition(depth_ref, rect);
    }

    tracked_array_msg.humans.push_back(human);
  }

  tracked_pub_->publish(tracked_array_msg);
}

}  // namespace opl_human_vision

RCLCPP_COMPONENTS_REGISTER_NODE(opl_human_vision::HumanTrackerNode)
