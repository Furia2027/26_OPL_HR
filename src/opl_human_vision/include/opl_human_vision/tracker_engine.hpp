#ifndef OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_
#define OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "opl_interfaces/msg/tracked_human.hpp"

namespace opl_human_vision {

struct Track {
  int id;
  cv::Rect2f bbox; // Switched to float precision to prevent truncation errors
  float confidence;
  int age{0};
  int hits{0};
  int time_since_update{0};
  std::string name{"unknown"};
};

class TrackerEngine {
public:
  TrackerEngine(int max_age = 30, int min_hits = 3, float iou_threshold = 0.3f);
  ~TrackerEngine() = default;

  std::vector<Track> update(const std::vector<opl_interfaces::msg::TrackedHuman>& detections, const cv::Mat& frame);

private:
  cv::Mat estimateCameraMotion(const cv::Mat& current_frame);
  float computeIoU(const cv::Rect2f& boxA, const cv::Rect2f& boxB);

  int max_age_;
  int min_hits_;
  float iou_threshold_;
  int next_id_{1};

  cv::Mat prev_frame_gray_;
  std::vector<Track> active_tracks_;
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_
