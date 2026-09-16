#pragma once

#include "opl_human_vision/scrfd_detector.hpp"
#include "opl_human_vision/adaface_extractor.hpp"
#include "opl_human_vision/osnet_extractor.hpp"

#include <geometry_msgs/msg/point.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace opl_human_vision {

struct FaceResult {
  uint64_t persistent_id = 0;
  std::string name = "Unknown";
  float match_score = 0.0f;
  bool just_scanned = false;

  std::vector<cv::Point2f> landmarks;
  std::vector<float> face_embedding;
  std::vector<float> body_embedding;
};


struct PersistentIdentity {
  uint64_t persistent_id = 0;
  std::string name;

  std::vector<std::vector<float>> face_gallery;
  std::vector<std::vector<float>> body_gallery;

  geometry_msgs::msg::Point last_position_3d;
  bool has_position_3d{false};

  std::chrono::steady_clock::time_point last_position_time{};

  bool has_face_embedding() const {
    return !face_gallery.empty();
  }

  bool has_body_embedding() const {
    return !body_gallery.empty();
  }
};


struct SessionIdentity {
  uint64_t session_id = 0;

  std::vector<std::vector<float>> face_gallery;
  std::vector<std::vector<float>> body_gallery;

  std::chrono::steady_clock::time_point last_seen;

  geometry_msgs::msg::Point last_position_3d;
  bool has_position_3d{false};

  std::chrono::steady_clock::time_point last_position_time{};

  bool has_face_embedding() const {
    return !face_gallery.empty();
  }

  bool has_body_embedding() const {
    return !body_gallery.empty();
  }
};


struct PendingEnrollment {
  int frame_count = 0;

  std::chrono::steady_clock::time_point first_seen;
  std::chrono::steady_clock::time_point last_seen;

  std::vector<std::vector<float>> face_embeddings;
  std::vector<std::vector<float>> body_embeddings;
};


class FaceEngine {
public:
  FaceEngine(
    const std::string& scrfd_path,
    const std::string& adaface_path,
    const std::string& osnet_path,
    float face_match_threshold = 0.50f,
    float body_match_threshold = 0.45f,
    int min_confirm_frames = 5,
    float min_confirm_duration_sec = 0.5f,
    double session_reid_timeout_sec = 5.0,
    double spatial_gate_max_age_sec = 2.0,
    float spatial_gate_base_m = 0.75f,
    float spatial_gate_max_speed_mps = 2.5f,
    float depth_gate_base_m = 0.75f
  );

  bool loadModels();

  FaceResult processFace(
    const cv::Mat& human_crop,
    uint64_t raw_track_id,
    const std::set<uint64_t>& occupied_ids,
    const geometry_msgs::msg::Point& position_3d,
    bool force_deep_reid = false
  );

  void cleanupStaleRawTracks(
    const std::set<uint64_t>& active_raw_ids
  );

  void authorizeEnrollment(
    uint64_t raw_track_id,
    const std::string& custom_name = "")
  {
    authorized_enrollments_.insert(raw_track_id);

    if (!custom_name.empty()) {
      custom_names_request_[raw_track_id] =
        custom_name;
    }
  }

  void updateIdentityPosition(
    uint64_t identity_id,
    const geometry_msgs::msg::Point& position_3d
  );

private:
  std::unique_ptr<ScrfdDetector> scrfd_;
  std::unique_ptr<AdaFaceExtractor> adaface_;
  std::unique_ptr<OsnetExtractor> osnet_;

  float face_match_threshold_;
  float body_match_threshold_;

  int min_confirm_frames_;
  float min_confirm_duration_sec_;

  double session_reid_timeout_sec_;

  double reid_spatial_gate_max_age_sec_{2.0};
  float reid_spatial_gate_base_m_{0.75f};
  float reid_spatial_gate_max_speed_mps_{2.5f};
  float reid_depth_gate_base_m_{0.75f};

  uint64_t next_persistent_id_{1};

  std::vector<PersistentIdentity>
    reid_gallery_;

  static constexpr uint64_t SESSION_ID_BASE =
    1000000000ULL;

  uint64_t next_session_id_{
    SESSION_ID_BASE
  };

  std::vector<SessionIdentity>
    session_gallery_;

  std::unordered_map<uint64_t, uint64_t>
    active_raw_to_persistent_map_;

  std::unordered_map<uint64_t, uint64_t>
    active_raw_to_session_map_;

  std::unordered_map<uint64_t, int>
    track_frame_counters_;

  std::unordered_map<uint64_t, PendingEnrollment>
    pending_enrollments_;

  std::set<uint64_t>
    authorized_enrollments_;

  std::unordered_map<uint64_t, std::string>
    custom_names_request_;

  std::chrono::steady_clock::time_point
    last_cleanup_time_;

  void cleanupStalePendingTracks();

  void cleanupStaleSessionIdentities();

  float getMaxCosineSimilarity(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& gallery
  );

  void addTemplateIfDiverse(
    std::vector<std::vector<float>>& gallery,
    const std::vector<float>& new_emb,
    size_t max_size,
    float thresh
  );

  std::vector<float> computeAverageEmbedding(
    const std::vector<std::vector<float>>& embeddings
  );

  void removeSessionIdentity(
    uint64_t session_id
  );

  bool isValid3DPosition(
    const geometry_msgs::msg::Point& p
  ) const;

  bool passesSpatialGate(
    const geometry_msgs::msg::Point& current,
    const geometry_msgs::msg::Point& previous,
    bool has_previous,
    const std::chrono::steady_clock::time_point& previous_time,
    const std::chrono::steady_clock::time_point& now
  ) const;

  void updateRecentPosition(
    const geometry_msgs::msg::Point& current,
    geometry_msgs::msg::Point& stored,
    bool& has_stored,
    std::chrono::steady_clock::time_point& stored_time,
    const std::chrono::steady_clock::time_point& now
  );
};


inline void drawFaceDebugOverlay(
  cv::Mat& frame,
  const cv::Rect& person_box,
  const FaceResult& result)
{
  if (
    person_box.width <= 0 ||
    person_box.height <= 0)
  {
    return;
  }

  cv::rectangle(
    frame,
    person_box,
    cv::Scalar(0, 255, 0),
    2
  );

  for (const auto& pt : result.landmarks) {
    cv::Point2f abs_pt(
      person_box.x + pt.x,
      person_box.y + pt.y
    );

    cv::circle(
      frame,
      abs_pt,
      2,
      cv::Scalar(0, 0, 255),
      -1
    );
  }

  std::string label =
    result.name +
    " (" +
    std::to_string(
      static_cast<int>(
        result.match_score * 100
      )
    ) +
    "%)";

  cv::putText(
    frame,
    label,
    cv::Point(
      person_box.x,
      std::max(
        0,
        person_box.y - 10
      )
    ),
    cv::FONT_HERSHEY_SIMPLEX,
    0.5,
    cv::Scalar(0, 255, 0),
    2
  );
}

}  // namespace opl_human_vision