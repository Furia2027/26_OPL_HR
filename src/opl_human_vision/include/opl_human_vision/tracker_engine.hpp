#ifndef OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_
#define OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "opl_interfaces/msg/tracked_human.hpp"

#include "opl_human_vision/osnet_extractor.hpp"

namespace opl_human_vision {


enum class TrackState {
  Tentative,
  Confirmed,
  Lost
};


struct Track {
  int id;
  cv::Rect2f bbox;
  float confidence;

  int age{0};

  // Total successful detector associations over this track's lifetime.
  int hits{0};

  // Consecutive successful detector associations.
  int hit_streak{0};

  int time_since_update{0};

  TrackState state{TrackState::Tentative};

  std::string name{"unknown"};

  // Pixels / second.
  cv::Point2f velocity{0.0f, 0.0f};

  // Last REAL detector observation, compensated into the current
  // camera coordinate frame by GMC while the track is lost.
  cv::Point2f last_observed_center{0.0f, 0.0f};

  bool velocity_initialized{false};

  // Timestamp corresponding to the latest prediction update.
  double last_prediction_time_sec{0.0};

  // Timestamp corresponding to the latest REAL detector observation.
  double last_observation_time_sec{0.0};
};


class TrackerEngine {
public:
  TrackerEngine(
    int max_age = 30,
    int min_hits = 3,
    float iou_threshold = 0.3f,
    int max_publish_age = 1,

    const std::string& osnet_model_path = "",
    bool appearance_reid_enabled = false,
    float appearance_match_threshold = 0.85f,
    int appearance_reid_max_age = 10,
    int appearance_refresh_interval = 15,
    float appearance_spatial_base_ratio = 0.10f,
    float appearance_spatial_speed_ratio_per_sec = 0.75f
  );

  ~TrackerEngine() = default;

  std::vector<Track> update(
    const std::vector<opl_interfaces::msg::TrackedHuman>& detections,
    const cv::Mat& frame,
    double timestamp_sec
  );

  bool appearanceReidReady() const {
    return
      appearance_reid_enabled_ &&
      appearance_reid_ready_;
  }

private:
  struct TrackAppearanceState {
    std::vector<std::vector<float>> body_gallery;

    int successful_matches_since_refresh{0};
  };


  cv::Mat estimateCameraMotion(
    const cv::Mat& current_frame
  );

  float computeIoU(
    const cv::Rect2f& boxA,
    const cv::Rect2f& boxB
  ) const;

  cv::Rect2f detectionBox(
    const opl_interfaces::msg::TrackedHuman& detection
  ) const;

  std::vector<float> extractAppearance(
    const cv::Mat& frame,
    const cv::Rect2f& bbox
  );

  float cosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b
  ) const;

  float getMaxAppearanceSimilarity(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& gallery
  ) const;

  void addAppearanceTemplate(
    TrackAppearanceState& appearance,
    const std::vector<float>& embedding
  );

  void maybeRefreshAppearance(
    const Track& track,
    const cv::Rect2f& observed_bbox,
    const cv::Mat& frame
  );

  bool passesAppearanceSpatialGate(
    const Track& track,
    const cv::Rect2f& detection_bbox,
    const cv::Mat& frame,
    double timestamp_sec
  ) const;

  void applyDetectionToTrack(
    Track& track,
    const opl_interfaces::msg::TrackedHuman& detection,
    const cv::Mat& frame,
    double timestamp_sec,
    bool refresh_appearance
  );


  float velocity_smoothing_{0.65f};

  int velocity_decay_after_{5};
  float velocity_decay_{0.95f};

  double nominal_dt_sec_{1.0 / 30.0};

  double max_prediction_dt_sec_{0.20};

  double min_valid_dt_sec_{1e-4};


  int max_age_;
  int min_hits_;
  float iou_threshold_;
  int max_publish_age_;


  // --------------------------------------------------------------------------
  // Appearance-assisted Lost-track recovery.
  // --------------------------------------------------------------------------

  bool appearance_reid_enabled_{false};
  bool appearance_reid_ready_{false};

  float appearance_match_threshold_{0.85f};

  int appearance_reid_max_age_{10};
  int appearance_refresh_interval_{15};

  // Maximum Stage-2 center displacement:
  //
  // allowed_px =
  //   image_scale *
  //   (
  //     base_ratio +
  //     speed_ratio_per_sec * elapsed_sec
  //   )
  //
  float appearance_spatial_base_ratio_{0.10f};

  float appearance_spatial_speed_ratio_per_sec_{0.75f};

  size_t appearance_gallery_max_size_{3};

  // Only store a new appearance template if it is sufficiently
  // different from the templates already in the track gallery.
  float appearance_gallery_diversity_threshold_{0.90f};

  std::unique_ptr<OsnetExtractor> appearance_osnet_;

  std::unordered_map<int, TrackAppearanceState>
    appearance_states_;


  int next_id_{1};

  cv::Mat prev_frame_gray_;

  std::vector<Track> active_tracks_;
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_