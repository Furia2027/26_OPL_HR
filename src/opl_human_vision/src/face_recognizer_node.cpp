#include "opl_human_vision/face_recognizer_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <geometry_msgs/msg/point.hpp>
#include <opl_interfaces/srv/enroll_person.hpp>
#include <algorithm>
#include <cv_bridge/cv_bridge.h>

namespace opl_human_vision {

FaceRecognizerNode::FaceRecognizerNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("face_recognizer_node", options) {
  declare_parameter<std::string>("image_topic", "/image_raw");
  declare_parameter<std::string>("tracked_topic", "tracked_humans");
  declare_parameter<std::string>("recognized_topic", "recognized_humans");
  declare_parameter<std::string>("scrfd_model_path", "");
  declare_parameter<std::string>("adaface_model_path", "");
  declare_parameter<std::string>("osnet_model_path", "");
  declare_parameter<float>("match_threshold", 0.45f);
  declare_parameter<float>("body_match_threshold", 0.55f);
  declare_parameter<int>("min_confirm_frames", 5);
  declare_parameter<float>("min_confirm_duration_sec", 0.5f);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "Configuring FaceRecognizerNode...");

  image_topic_ = get_parameter("image_topic").as_string();
  tracked_topic_ = get_parameter("tracked_topic").as_string();
  recognized_topic_ = get_parameter("recognized_topic").as_string();

  std::string scrfd_path = get_parameter("scrfd_model_path").as_string();
  std::string adaface_path = get_parameter("adaface_model_path").as_string();
  std::string osnet_path = get_parameter("osnet_model_path").as_string();
  float match_thresh = static_cast<float>(get_parameter("match_threshold").as_double());
  float body_thresh = static_cast<float>(get_parameter("body_match_threshold").as_double());
  int min_confirm_frames = get_parameter("min_confirm_frames").as_int();
  float min_confirm_duration = static_cast<float>(get_parameter("min_confirm_duration_sec").as_double());

  face_engine_ = std::make_shared<FaceEngine>(
    scrfd_path, adaface_path, osnet_path,
    match_thresh, body_thresh,
    min_confirm_frames, min_confirm_duration
  );

  if (!face_engine_->loadModels()) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize TensorRT models in FaceEngine!");
    return CallbackReturn::FAILURE;
  }

  // Publisher setup
  recognized_pub_ = create_publisher<opl_interfaces::msg::TrackedHumanArray>(recognized_topic_, rclcpp::SensorDataQoS());

  // Service setup for triggers
  enroll_srv_ = create_service<opl_interfaces::srv::EnrollPerson>(
    "~/enroll_person",
    std::bind(&FaceRecognizerNode::handleEnrollService, this, std::placeholders::_1, std::placeholders::_2)
  );

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_activate(const rclcpp_lifecycle::State & state) {
  LifecycleNode::on_activate(state);
  recognized_pub_->on_activate();

  image_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  tracked_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions img_sub_options;
  img_sub_options.callback_group = image_cb_group_;
  img_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    image_topic_, rclcpp::SensorDataQoS(),
    std::bind(&FaceRecognizerNode::imageCallback, this, std::placeholders::_1),
    img_sub_options
  );

  rclcpp::SubscriptionOptions tracked_sub_options;
  tracked_sub_options.callback_group = tracked_cb_group_;
  tracked_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  tracked_sub_ = create_subscription<opl_interfaces::msg::TrackedHumanArray>(
    tracked_topic_, rclcpp::SensorDataQoS(),
    std::bind(&FaceRecognizerNode::trackedHumansCallback, this, std::placeholders::_1),
    tracked_sub_options
  );

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_deactivate(const rclcpp_lifecycle::State & state) {
  LifecycleNode::on_deactivate(state);
  image_sub_.reset();
  tracked_sub_.reset();
  recognized_pub_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_cleanup(const rclcpp_lifecycle::State &) {
  recognized_pub_.reset();
  enroll_srv_.reset();
  face_engine_.reset();

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    currently_visible_pids_.clear();
    ever_scanned_pids_.clear();
    pid_to_name_.clear();
    locked_tracks_.clear();
  }

  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frame_buffer_.clear();
  }

  return CallbackReturn::SUCCESS;
}

void FaceRecognizerNode::handleEnrollService(
  const std::shared_ptr<opl_interfaces::srv::EnrollPerson::Request> request,
  std::shared_ptr<opl_interfaces::srv::EnrollPerson::Response> response)
{
  uint64_t target_id = request->raw_track_id;

  // Auto-select nearest unassigned track if ID is 0
  if (target_id == 0) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!last_active_raw_ids_.empty()) {
      target_id = *last_active_raw_ids_.begin(); // Fallback to first active unassigned track
    }
  }

  if (target_id == 0) {
    response->success = false;
    response->message = "No active track found to authorize enrollment.";
    return;
  }

  {
    std::lock_guard<std::mutex> gpu_lock(engine_mutex_);
    face_engine_->authorizeEnrollment(target_id, request->name);
  }

  response->success = true;
  response->message = "Enrollment authorized for Raw Track ID " + std::to_string(target_id);
  RCLCPP_INFO(get_logger(), "🔓 Service trigger received: Enrollment authorized for track %lu", target_id);
}

void FaceRecognizerNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  try {
    auto cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frame_buffer_.emplace_back(rclcpp::Time(msg->header.stamp), cv_ptr->image);

    while (frame_buffer_.size() > 30) {
      frame_buffer_.pop_front();
    }
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
  }
}

void FaceRecognizerNode::trackedHumansCallback(const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg) {
  if (!recognized_pub_->is_activated()) return;

  cv::Mat frame_ref;
  rclcpp::Time target_time(msg->header.stamp);
  double min_diff = 1e9;

  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (frame_buffer_.empty()) {
      recognized_pub_->publish(*msg);
      return;
    }

    while (!frame_buffer_.empty() && (target_time - frame_buffer_.front().first).seconds() > 0.20) {
      frame_buffer_.pop_front();
    }

    for (const auto& item : frame_buffer_) {
      double diff = std::abs((item.first - target_time).seconds());
      if (diff < min_diff) {
        min_diff = diff;
        frame_ref = item.second;
      }
    }
  }

  if (frame_ref.empty() || min_diff > 0.10) {
    recognized_pub_->publish(*msg);
    return;
  }

  opl_interfaces::msg::TrackedHumanArray recognized_array_msg;
  recognized_array_msg.header = msg->header;

  std::set<uint64_t> assigned_ids_in_frame;
  std::set<uint64_t> current_frame_pids;
  std::set<uint64_t> active_raw_ids;

  for (const auto& human : msg->humans) {
    active_raw_ids.insert(human.track_id);
  }

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    last_active_raw_ids_ = active_raw_ids;
  }

  {
    std::lock_guard<std::mutex> gpu_lock(engine_mutex_);
    face_engine_->cleanupStaleRawTracks(active_raw_ids);
  }

  // Handle Unlocking when a temporary track ID leaves screen
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    for (auto it = locked_tracks_.begin(); it != locked_tracks_.end(); ) {
      if (active_raw_ids.find(it->first) == active_raw_ids.end()) {
        RCLCPP_INFO(get_logger(), "🔓 Person left screen. Unlocking Permanent ID %lu ('%s') from Temp Track %lu.",
                    it->second.persistent_id, it->second.name.c_str(), it->first);
        it = locked_tracks_.erase(it);
      } else {
        ++it;
      }
    }
  }

  for (auto human : msg->humans) {
    uint64_t raw_track_id = human.track_id;
    bool is_locked = false;
    LockedTrackInfo locked_info;

    {
      std::lock_guard<std::mutex> state_lock(state_mutex_);
      auto it = locked_tracks_.find(raw_track_id);
      if (it != locked_tracks_.end()) {
        is_locked = true;
        locked_info = it->second;
      }
    }

    if (is_locked) {
      // FAST PATH: Identity is locked
      human.track_id = locked_info.persistent_id;
      human.name = locked_info.name;

      assigned_ids_in_frame.insert(locked_info.persistent_id);
      current_frame_pids.insert(locked_info.persistent_id);
    } else {
      // SLOW PATH: Unknown or newly entered track
      int raw_x = static_cast<int>(human.bbox.x_offset);
      int raw_y = static_cast<int>(human.bbox.y_offset);
      int raw_w = static_cast<int>(human.bbox.width);
      int raw_h = static_cast<int>(human.bbox.height);

      int x1 = std::max(0, std::min(raw_x, frame_ref.cols - 1));
      int y1 = std::max(0, std::min(raw_y, frame_ref.rows - 1));
      int x2 = std::max(0, std::min(raw_x + raw_w, frame_ref.cols));
      int y2 = std::max(0, std::min(raw_y + raw_h, frame_ref.rows));

      int crop_w = x2 - x1;
      int crop_h = y2 - y1;

      if (crop_w > 0 && crop_h > 0) {
        cv::Rect human_rect(x1, y1, crop_w, crop_h);
        cv::Mat human_crop = frame_ref(human_rect).clone();

        FaceResult result;
        {
          std::lock_guard<std::mutex> gpu_lock(engine_mutex_);
          result = face_engine_->processFace(human_crop, raw_track_id, assigned_ids_in_frame);
        }

        human.landmarks.clear();
        for (const auto& lm : result.landmarks) {
          geometry_msgs::msg::Point pt;
          pt.x = static_cast<double>(x1) + lm.x;
          pt.y = static_cast<double>(y1) + lm.y;
          pt.z = 0.0;
          human.landmarks.push_back(pt);
        }

        if (result.persistent_id > 0) {
          human.track_id = result.persistent_id;
          human.name = result.name;
          assigned_ids_in_frame.insert(result.persistent_id);
          current_frame_pids.insert(result.persistent_id);

          {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            pid_to_name_[result.persistent_id] = result.name;
            if (result.just_scanned) {
              ever_scanned_pids_.insert(result.persistent_id);
            }
          }
        } else {
          human.name = result.name;
        }

        // CHECK LOCK CONDITION
        std::string name_lower = result.name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        bool is_scanning = (name_lower.find("scanning") != std::string::npos);
        bool is_unregistered = (name_lower.find("unregistered") != std::string::npos);

        if (!result.name.empty() && !is_scanning && !is_unregistered && result.persistent_id > 0) {
          std::lock_guard<std::mutex> state_lock(state_mutex_);
          LockedTrackInfo new_lock;
          new_lock.name = result.name;
          new_lock.persistent_id = result.persistent_id;
          locked_tracks_[raw_track_id] = new_lock;

          RCLCPP_INFO(get_logger(), "🔒 Identity Locked: Mapped Temp Track %lu -> Permanent ID %lu ('%s')",
                      raw_track_id, result.persistent_id, result.name.c_str());
        }
      }
    }
    recognized_array_msg.humans.push_back(human);
  }

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    currently_visible_pids_ = current_frame_pids;
  }

  recognized_pub_->publish(recognized_array_msg);
}

}  // namespace opl_human_vision

RCLCPP_COMPONENTS_REGISTER_NODE(opl_human_vision::FaceRecognizerNode)
