#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opl_interfaces/msg/tracked_human_array.hpp>
#include <opl_interfaces/srv/enroll_person.hpp>
#include "opl_human_vision/face_engine.hpp"

#include <mutex>
#include <deque>
#include <unordered_map>
#include <set>

namespace opl_human_vision {

struct LockedTrackInfo {
  std::string name;
  uint64_t persistent_id = 0;
};

class FaceRecognizerNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit FaceRecognizerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

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
  void trackedHumansCallback(const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg);
  void handleEnrollService(
    const std::shared_ptr<opl_interfaces::srv::EnrollPerson::Request> request,
    std::shared_ptr<opl_interfaces::srv::EnrollPerson::Response> response);

  std::string image_topic_;
  std::string tracked_topic_;
  std::string recognized_topic_;

  rclcpp::CallbackGroup::SharedPtr image_cb_group_;
  rclcpp::CallbackGroup::SharedPtr tracked_cb_group_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<opl_interfaces::msg::TrackedHumanArray>::SharedPtr tracked_sub_;
  rclcpp_lifecycle::LifecyclePublisher<opl_interfaces::msg::TrackedHumanArray>::SharedPtr recognized_pub_;
  rclcpp::Service<opl_interfaces::srv::EnrollPerson>::SharedPtr enroll_srv_;

  std::shared_ptr<FaceEngine> face_engine_;
  std::mutex engine_mutex_;

  std::mutex frame_mutex_;
  std::deque<std::pair<rclcpp::Time, cv::Mat>> frame_buffer_;

  std::mutex state_mutex_;
  std::unordered_map<uint64_t, LockedTrackInfo> locked_tracks_;
  std::unordered_map<uint64_t, std::string> pid_to_name_;
  std::set<uint64_t> currently_visible_pids_;
  std::set<uint64_t> ever_scanned_pids_;
  std::set<uint64_t> last_active_raw_ids_;
};

} // namespace opl_human_vision
