#ifndef OPL_HUMAN_VISION__HUMAN_TRACKER_NODE_HPP_
#define OPL_HUMAN_VISION__HUMAN_TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>

#include "opl_interfaces/msg/tracked_human_array.hpp"
#include "opl_human_vision/tracker_engine.hpp"

namespace opl_human_vision {

class HumanTrackerNode
  : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit HumanTrackerNode(
    const rclcpp::NodeOptions & options =
      rclcpp::NodeOptions()
  );

  ~HumanTrackerNode() override =
    default;


  rclcpp_lifecycle::node_interfaces::
    LifecycleNodeInterface::CallbackReturn
  on_configure(
    const rclcpp_lifecycle::State &
  ) override;


  rclcpp_lifecycle::node_interfaces::
    LifecycleNodeInterface::CallbackReturn
  on_activate(
    const rclcpp_lifecycle::State &
  ) override;


  rclcpp_lifecycle::node_interfaces::
    LifecycleNodeInterface::CallbackReturn
  on_deactivate(
    const rclcpp_lifecycle::State &
  ) override;


  rclcpp_lifecycle::node_interfaces::
    LifecycleNodeInterface::CallbackReturn
  on_cleanup(
    const rclcpp_lifecycle::State &
  ) override;


private:
  void imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg
  );

  void depthCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg
  );

  void cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg
  );

  void detectionsCallback(
    const opl_interfaces::msg::
      TrackedHumanArray::ConstSharedPtr msg
  );


  geometry_msgs::msg::Point compute3DPosition(
    const cv::Mat& depth_img,
    const cv::Rect& bbox
  );


  std::string image_topic_;
  std::string depth_topic_;
  std::string detections_topic_;
  std::string tracked_topic_;


  rclcpp::CallbackGroup::SharedPtr
    image_cb_group_;

  rclcpp::CallbackGroup::SharedPtr
    depth_cb_group_;

  rclcpp::CallbackGroup::SharedPtr
    track_cb_group_;


  rclcpp::Subscription<
    sensor_msgs::msg::Image
  >::SharedPtr image_sub_;


  rclcpp::Subscription<
    sensor_msgs::msg::Image
  >::SharedPtr depth_sub_;


  rclcpp::Subscription<
    sensor_msgs::msg::CameraInfo
  >::SharedPtr info_sub_;


  rclcpp::Subscription<
    opl_interfaces::msg::TrackedHumanArray
  >::SharedPtr detection_sub_;


  rclcpp_lifecycle::LifecyclePublisher<
    opl_interfaces::msg::TrackedHumanArray
  >::SharedPtr tracked_pub_;


  std::shared_ptr<TrackerEngine>
    tracker_;


  std::mutex frame_mutex_;

  cv_bridge::CvImageConstPtr
    latest_frame_ptr_;


  std::mutex depth_mutex_;

  cv_bridge::CvImageConstPtr
    latest_depth_ptr_;


  std::mutex camera_info_mutex_;

  float fx_{0.0f};
  float fy_{0.0f};
  float cx_{0.0f};
  float cy_{0.0f};

  bool has_camera_info_{false};
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__HUMAN_TRACKER_NODE_HPP_