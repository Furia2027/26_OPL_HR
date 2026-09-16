#include "opl_human_vision/tracker_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>


namespace {

// -----------------------------------------------------------------------------
// Hungarian minimum-cost assignment solver.
//
// Input:
//   cost[track][detection]
//
// Output:
//   assignment[track] = detection
//
// -1 means unassigned.
// -----------------------------------------------------------------------------

std::vector<int> solveHungarian(
  const std::vector<std::vector<float>>& input_cost)
{
  const std::size_t original_rows =
    input_cost.size();

  if (original_rows == 0) {
    return {};
  }


  const std::size_t original_cols =
    input_cost.front().size();


  std::vector<int> result(
    original_rows,
    -1
  );


  if (original_cols == 0) {
    return result;
  }


  const bool transpose =
    original_rows >
    original_cols;


  std::vector<std::vector<double>> cost;


  if (!transpose) {

    cost.resize(
      original_rows,
      std::vector<double>(
        original_cols,
        0.0
      )
    );


    for (
      std::size_t i = 0;
      i < original_rows;
      ++i)
    {
      for (
        std::size_t j = 0;
        j < original_cols;
        ++j)
      {
        cost[i][j] =
          static_cast<double>(
            input_cost[i][j]
          );
      }
    }

  } else {

    cost.resize(
      original_cols,
      std::vector<double>(
        original_rows,
        0.0
      )
    );


    for (
      std::size_t i = 0;
      i < original_rows;
      ++i)
    {
      for (
        std::size_t j = 0;
        j < original_cols;
        ++j)
      {
        cost[j][i] =
          static_cast<double>(
            input_cost[i][j]
          );
      }
    }
  }


  const int n =
    static_cast<int>(
      cost.size()
    );

  const int m =
    static_cast<int>(
      cost.front().size()
    );


  std::vector<double> u(
    n + 1,
    0.0
  );

  std::vector<double> v(
    m + 1,
    0.0
  );

  std::vector<int> p(
    m + 1,
    0
  );

  std::vector<int> way(
    m + 1,
    0
  );


  for (int i = 1; i <= n; ++i) {

    p[0] =
      i;


    int j0 =
      0;


    std::vector<double> minv(
      m + 1,
      std::numeric_limits<double>::infinity()
    );

    std::vector<bool> used(
      m + 1,
      false
    );


    do {

      used[j0] =
        true;


      const int i0 =
        p[j0];


      double delta =
        std::numeric_limits<double>::infinity();


      int j1 =
        0;


      for (int j = 1; j <= m; ++j) {

        if (used[j]) {
          continue;
        }


        const double current_cost =
          cost[i0 - 1][j - 1] -
          u[i0] -
          v[j];


        if (
          current_cost <
          minv[j])
        {
          minv[j] =
            current_cost;

          way[j] =
            j0;
        }


        if (
          minv[j] <
          delta)
        {
          delta =
            minv[j];

          j1 =
            j;
        }
      }


      for (int j = 0; j <= m; ++j) {

        if (used[j]) {

          u[p[j]] +=
            delta;

          v[j] -=
            delta;

        } else {

          minv[j] -=
            delta;
        }
      }


      j0 =
        j1;

    } while (
      p[j0] != 0
    );


    do {

      const int j1 =
        way[j0];


      p[j0] =
        p[j1];


      j0 =
        j1;

    } while (
      j0 != 0
    );
  }


  if (!transpose) {

    for (int j = 1; j <= m; ++j) {

      if (p[j] == 0) {
        continue;
      }


      const int track_index =
        p[j] - 1;

      const int detection_index =
        j - 1;


      if (
        track_index >= 0 &&
        track_index <
          static_cast<int>(
            original_rows
          ))
      {
        result[track_index] =
          detection_index;
      }
    }

  } else {

    for (int j = 1; j <= m; ++j) {

      if (p[j] == 0) {
        continue;
      }


      const int original_detection =
        p[j] - 1;

      const int original_track =
        j - 1;


      if (
        original_track >= 0 &&
        original_track <
          static_cast<int>(
            original_rows
          ) &&
        original_detection >= 0 &&
        original_detection <
          static_cast<int>(
            original_cols
          ))
      {
        result[original_track] =
          original_detection;
      }
    }
  }


  return result;
}

}  // namespace


namespace opl_human_vision {


TrackerEngine::TrackerEngine(
  int max_age,
  int min_hits,
  float iou_threshold,
  int max_publish_age,
  const std::string& osnet_model_path,
  bool appearance_reid_enabled,
  float appearance_match_threshold,
  int appearance_reid_max_age,
  int appearance_refresh_interval,
  float appearance_spatial_base_ratio,
  float appearance_spatial_speed_ratio_per_sec)
: max_age_(max_age),
  min_hits_(min_hits),
  iou_threshold_(iou_threshold),
  max_publish_age_(max_publish_age),
  appearance_reid_enabled_(appearance_reid_enabled),
  appearance_match_threshold_(appearance_match_threshold),
  appearance_reid_max_age_(appearance_reid_max_age),
  appearance_refresh_interval_(appearance_refresh_interval),
  appearance_spatial_base_ratio_(appearance_spatial_base_ratio),
  appearance_spatial_speed_ratio_per_sec_(
    appearance_spatial_speed_ratio_per_sec)
{
  appearance_match_threshold_ =
    std::max(
      -1.0f,
      std::min(
        1.0f,
        appearance_match_threshold_
      )
    );


  appearance_reid_max_age_ =
    std::max(
      0,
      std::min(
        appearance_reid_max_age_,
        max_age_
      )
    );


  appearance_refresh_interval_ =
    std::max(
      1,
      appearance_refresh_interval_
    );


  appearance_spatial_base_ratio_ =
    std::max(
      0.0f,
      appearance_spatial_base_ratio_
    );


  appearance_spatial_speed_ratio_per_sec_ =
    std::max(
      0.0f,
      appearance_spatial_speed_ratio_per_sec_
    );


  if (
    appearance_reid_enabled_ &&
    !osnet_model_path.empty())
  {
    appearance_osnet_ =
      std::make_unique<OsnetExtractor>(
        osnet_model_path
      );


    appearance_reid_ready_ =
      appearance_osnet_->init();
  }
}


// -----------------------------------------------------------------------------
// Convert detector message bbox to OpenCV rectangle.
// -----------------------------------------------------------------------------

cv::Rect2f TrackerEngine::detectionBox(
  const opl_interfaces::msg::TrackedHuman& detection) const
{
  return cv::Rect2f(
    static_cast<float>(
      detection.bbox.x_offset
    ),
    static_cast<float>(
      detection.bbox.y_offset
    ),
    static_cast<float>(
      detection.bbox.width
    ),
    static_cast<float>(
      detection.bbox.height
    )
  );
}


// -----------------------------------------------------------------------------
// Global camera-motion estimation.
// -----------------------------------------------------------------------------

cv::Mat TrackerEngine::estimateCameraMotion(
  const cv::Mat& current_frame)
{
  cv::Mat transform =
    cv::Mat::eye(
      2,
      3,
      CV_64F
    );


  if (current_frame.empty()) {
    return transform;
  }


  cv::Mat gray;
  cv::Mat small_gray;


  if (
    current_frame.channels() ==
    3)
  {
    cv::cvtColor(
      current_frame,
      gray,
      cv::COLOR_BGR2GRAY
    );

  } else {

    gray =
      current_frame;
  }


  float scale =
    1.0f;


  if (gray.cols > 320) {

    scale =
      320.0f /
      static_cast<float>(
        gray.cols
      );


    cv::resize(
      gray,
      small_gray,
      cv::Size(),
      scale,
      scale,
      cv::INTER_NEAREST
    );

  } else {

    small_gray =
      gray;
  }


  if (
    prev_frame_gray_.empty())
  {
    prev_frame_gray_ =
      small_gray.clone();

    return transform;
  }


  cv::Mat mask =
    cv::Mat::ones(
      small_gray.size(),
      CV_8UC1
    ) * 255;


  for (
    const auto& track :
    active_tracks_)
  {
    const cv::Rect2f scaled_box(
      track.bbox.x * scale,
      track.bbox.y * scale,
      track.bbox.width * scale,
      track.bbox.height * scale
    );


    const int x1 =
      std::max(
        0,
        static_cast<int>(
          std::floor(
            scaled_box.x
          )
        )
      );


    const int y1 =
      std::max(
        0,
        static_cast<int>(
          std::floor(
            scaled_box.y
          )
        )
      );


    const int x2 =
      std::min(
        small_gray.cols,
        static_cast<int>(
          std::ceil(
            scaled_box.x +
            scaled_box.width
          )
        )
      );


    const int y2 =
      std::min(
        small_gray.rows,
        static_cast<int>(
          std::ceil(
            scaled_box.y +
            scaled_box.height
          )
        )
      );


    if (
      x2 > x1 &&
      y2 > y1)
    {
      mask(
        cv::Rect(
          x1,
          y1,
          x2 - x1,
          y2 - y1
        )
      ).setTo(0);
    }
  }


  std::vector<cv::Point2f>
    prev_pts;

  std::vector<cv::Point2f>
    curr_pts;


  cv::goodFeaturesToTrack(
    prev_frame_gray_,
    prev_pts,
    200,
    0.01,
    10,
    mask
  );


  if (
    prev_pts.size() >= 10)
  {
    std::vector<uchar>
      status;

    std::vector<float>
      err;


    cv::calcOpticalFlowPyrLK(
      prev_frame_gray_,
      small_gray,
      prev_pts,
      curr_pts,
      status,
      err
    );


    std::vector<cv::Point2f>
      valid_prev;

    std::vector<cv::Point2f>
      valid_curr;


    for (
      std::size_t i = 0;
      i < status.size();
      ++i)
    {
      if (status[i]) {

        valid_prev.push_back(
          prev_pts[i]
        );

        valid_curr.push_back(
          curr_pts[i]
        );
      }
    }


    if (
      valid_prev.size() >= 6)
    {
      cv::Mat small_transform =
        cv::estimateAffinePartial2D(
          valid_prev,
          valid_curr
        );


      if (
        !small_transform.empty() &&
        cv::checkRange(
          small_transform
        ))
      {
        small_transform.at<double>(
          0,
          2
        ) /= scale;


        small_transform.at<double>(
          1,
          2
        ) /= scale;


        transform =
          small_transform;
      }
    }
  }


  prev_frame_gray_ =
    small_gray.clone();


  return transform;
}


// -----------------------------------------------------------------------------
// IoU.
// -----------------------------------------------------------------------------

float TrackerEngine::computeIoU(
  const cv::Rect2f& boxA,
  const cv::Rect2f& boxB) const
{
  const float x1 =
    std::max(
      boxA.x,
      boxB.x
    );

  const float y1 =
    std::max(
      boxA.y,
      boxB.y
    );

  const float x2 =
    std::min(
      boxA.x +
        boxA.width,
      boxB.x +
        boxB.width
    );

  const float y2 =
    std::min(
      boxA.y +
        boxA.height,
      boxB.y +
        boxB.height
    );


  const float inter_area =
    std::max(
      0.0f,
      x2 - x1
    ) *
    std::max(
      0.0f,
      y2 - y1
    );


  const float box_a_area =
    boxA.width *
    boxA.height;

  const float box_b_area =
    boxB.width *
    boxB.height;


  return
    inter_area /
    (
      box_a_area +
      box_b_area -
      inter_area +
      1e-6f
    );
}


// -----------------------------------------------------------------------------
// Extract OSNet body embedding from a bbox.
// -----------------------------------------------------------------------------

std::vector<float> TrackerEngine::extractAppearance(
  const cv::Mat& frame,
  const cv::Rect2f& bbox)
{
  if (
    !appearanceReidReady() ||
    frame.empty())
  {
    return {};
  }


  const int x1 =
    std::max(
      0,
      std::min(
        static_cast<int>(
          std::floor(
            bbox.x
          )
        ),
        frame.cols
      )
    );


  const int y1 =
    std::max(
      0,
      std::min(
        static_cast<int>(
          std::floor(
            bbox.y
          )
        ),
        frame.rows
      )
    );


  const int x2 =
    std::max(
      0,
      std::min(
        static_cast<int>(
          std::ceil(
            bbox.x +
            bbox.width
          )
        ),
        frame.cols
      )
    );


  const int y2 =
    std::max(
      0,
      std::min(
        static_cast<int>(
          std::ceil(
            bbox.y +
            bbox.height
          )
        ),
        frame.rows
      )
    );


  const int width =
    x2 - x1;

  const int height =
    y2 - y1;


  if (
    width < 16 ||
    height < 32)
  {
    return {};
  }


  const cv::Mat person_crop =
    frame(
      cv::Rect(
        x1,
        y1,
        width,
        height
      )
    ).clone();


  return
    appearance_osnet_->extract(
      person_crop
    );
}


// -----------------------------------------------------------------------------
// Safe cosine similarity.
// -----------------------------------------------------------------------------

float TrackerEngine::cosineSimilarity(
  const std::vector<float>& a,
  const std::vector<float>& b) const
{
  if (
    a.empty() ||
    b.empty() ||
    a.size() != b.size())
  {
    return -1.0f;
  }


  double dot =
    0.0;

  double norm_a =
    0.0;

  double norm_b =
    0.0;


  for (
    std::size_t i = 0;
    i < a.size();
    ++i)
  {
    dot +=
      static_cast<double>(
        a[i]
      ) *
      static_cast<double>(
        b[i]
      );


    norm_a +=
      static_cast<double>(
        a[i]
      ) *
      static_cast<double>(
        a[i]
      );


    norm_b +=
      static_cast<double>(
        b[i]
      ) *
      static_cast<double>(
        b[i]
      );
  }


  if (
    norm_a < 1e-12 ||
    norm_b < 1e-12)
  {
    return -1.0f;
  }


  const double similarity =
    dot /
    (
      std::sqrt(norm_a) *
      std::sqrt(norm_b)
    );


  return
    static_cast<float>(
      std::max(
        -1.0,
        std::min(
          1.0,
          similarity
        )
      )
    );
}


// -----------------------------------------------------------------------------
// Best similarity to an appearance gallery.
// -----------------------------------------------------------------------------

float TrackerEngine::getMaxAppearanceSimilarity(
  const std::vector<float>& query,
  const std::vector<std::vector<float>>& gallery) const
{
  if (
    query.empty() ||
    gallery.empty())
  {
    return -1.0f;
  }


  float best =
    -1.0f;


  for (
    const auto& gallery_embedding :
    gallery)
  {
    best =
      std::max(
        best,
        cosineSimilarity(
          query,
          gallery_embedding
        )
      );
  }


  return best;
}


// -----------------------------------------------------------------------------
// Add a diverse body template without allowing the gallery to grow forever.
// -----------------------------------------------------------------------------

void TrackerEngine::addAppearanceTemplate(
  TrackAppearanceState& appearance,
  const std::vector<float>& embedding)
{
  if (embedding.empty()) {
    return;
  }


  if (
    appearance.body_gallery.empty())
  {
    appearance.body_gallery.push_back(
      embedding
    );

    return;
  }


  const float similarity =
    getMaxAppearanceSimilarity(
      embedding,
      appearance.body_gallery
    );


  if (
    similarity >=
    appearance_gallery_diversity_threshold_)
  {
    return;
  }


  if (
    appearance.body_gallery.size() >=
    appearance_gallery_max_size_)
  {
    appearance.body_gallery.erase(
      appearance.body_gallery.begin()
    );
  }


  appearance.body_gallery.push_back(
    embedding
  );
}


// -----------------------------------------------------------------------------
// Seed / periodically refresh a confirmed track's OSNet gallery.
//
// This is intentionally low-cadence. OSNet is NOT run every normal frame.
// -----------------------------------------------------------------------------

void TrackerEngine::maybeRefreshAppearance(
  const Track& track,
  const cv::Rect2f& observed_bbox,
  const cv::Mat& frame)
{
  if (
    !appearanceReidReady() ||
    track.state !=
      TrackState::Confirmed ||
    frame.empty())
  {
    return;
  }


  auto state_it =
    appearance_states_.find(
      track.id
    );


  // First time a confirmed track gets an appearance state:
  // seed immediately.
  if (
    state_it ==
    appearance_states_.end())
  {
    TrackAppearanceState
      new_state;


    const std::vector<float> embedding =
      extractAppearance(
        frame,
        observed_bbox
      );


    if (!embedding.empty()) {

      new_state.body_gallery.push_back(
        embedding
      );
    }


    appearance_states_[
      track.id
    ] =
      std::move(
        new_state
      );


    return;
  }


  TrackAppearanceState& appearance =
    state_it->second;


  appearance.successful_matches_since_refresh++;


  if (
    appearance.successful_matches_since_refresh <
    appearance_refresh_interval_)
  {
    return;
  }


  appearance.successful_matches_since_refresh =
    0;


  const std::vector<float> embedding =
    extractAppearance(
      frame,
      observed_bbox
    );


  addAppearanceTemplate(
    appearance,
    embedding
  );
}


// -----------------------------------------------------------------------------
// Stage-2 2D plausibility gate.
//
// The predicted bbox has already been updated by GMC + velocity prediction.
//
// allowed_distance grows with real elapsed time:
//   image_scale * (base_ratio + speed_ratio * dt)
//
// This is only a gate. Appearance still decides the assignment.
// -----------------------------------------------------------------------------

bool TrackerEngine::passesAppearanceSpatialGate(
  const Track& track,
  const cv::Rect2f& detection_bbox,
  const cv::Mat& frame,
  double timestamp_sec) const
{
  if (frame.empty()) {
    return false;
  }


  const cv::Point2f predicted_center(
    track.bbox.x +
      track.bbox.width *
      0.5f,

    track.bbox.y +
      track.bbox.height *
      0.5f
  );


  const cv::Point2f detection_center(
    detection_bbox.x +
      detection_bbox.width *
      0.5f,

    detection_bbox.y +
      detection_bbox.height *
      0.5f
  );


  const float dx =
    detection_center.x -
    predicted_center.x;

  const float dy =
    detection_center.y -
    predicted_center.y;


  const float distance =
    std::sqrt(
      dx * dx +
      dy * dy
    );


  double elapsed_sec =
    nominal_dt_sec_ *
    static_cast<double>(
      std::max(
        1,
        track.time_since_update
      )
    );


  const bool timestamp_valid =
    std::isfinite(
      timestamp_sec
    ) &&
    timestamp_sec > 0.0;


  if (
    timestamp_valid &&
    track.last_observation_time_sec > 0.0)
  {
    const double measured_elapsed =
      timestamp_sec -
      track.last_observation_time_sec;


    if (
      measured_elapsed >= 0.0)
    {
      elapsed_sec =
        measured_elapsed;
    }
  }


  const float frame_scale =
    static_cast<float>(
      std::max(
        frame.cols,
        frame.rows
      )
    );


  float allowed_ratio =
    appearance_spatial_base_ratio_ +
    appearance_spatial_speed_ratio_per_sec_ *
      static_cast<float>(
        elapsed_sec
      );


  // Do not let the recovery gate become effectively infinite.
  allowed_ratio =
    std::min(
      1.0f,
      std::max(
        appearance_spatial_base_ratio_,
        allowed_ratio
      )
    );


  const float allowed_distance =
    frame_scale *
    allowed_ratio;


  return
    distance <=
    allowed_distance;
}


// -----------------------------------------------------------------------------
// Shared detector -> track update.
//
// Both Stage 1 IoU matches and Stage 2 appearance recovery use this function,
// so velocity/state/lifecycle behavior remains identical.
// -----------------------------------------------------------------------------

void TrackerEngine::applyDetectionToTrack(
  Track& track,
  const opl_interfaces::msg::TrackedHuman& detection,
  const cv::Mat& frame,
  double timestamp_sec,
  bool refresh_appearance)
{
  const cv::Rect2f det_box =
    detectionBox(
      detection
    );


  const cv::Point2f det_center(
    det_box.x +
      det_box.width *
      0.5f,

    det_box.y +
      det_box.height *
      0.5f
  );


  const int elapsed_updates =
    std::max(
      1,
      track.time_since_update
    );


  double observation_dt =
    nominal_dt_sec_ *
    static_cast<double>(
      elapsed_updates
    );


  const bool timestamp_valid =
    std::isfinite(
      timestamp_sec
    ) &&
    timestamp_sec > 0.0;


  if (
    timestamp_valid &&
    track.last_observation_time_sec > 0.0)
  {
    const double measured_dt =
      timestamp_sec -
      track.last_observation_time_sec;


    if (
      measured_dt >
      min_valid_dt_sec_)
    {
      observation_dt =
        measured_dt;
    }
  }


  observation_dt =
    std::max(
      observation_dt,
      min_valid_dt_sec_
    );


  cv::Point2f measured_velocity(
    static_cast<float>(
      (
        det_center.x -
        track.last_observed_center.x
      ) /
      observation_dt
    ),

    static_cast<float>(
      (
        det_center.y -
        track.last_observed_center.y
      ) /
      observation_dt
    )
  );


  // Preserve existing velocity safety behavior, now in px/s.
  if (!frame.empty()) {

    const float old_per_update_limit =
      0.25f *
      static_cast<float>(
        std::max(
          frame.cols,
          frame.rows
        )
      );


    const float max_velocity_per_sec =
      old_per_update_limit /
      static_cast<float>(
        nominal_dt_sec_
      );


    const float speed =
      std::sqrt(
        measured_velocity.x *
          measured_velocity.x +
        measured_velocity.y *
          measured_velocity.y
      );


    if (
      speed >
        max_velocity_per_sec &&
      speed > 1e-6f)
    {
      const float scale =
        max_velocity_per_sec /
        speed;


      measured_velocity.x *=
        scale;

      measured_velocity.y *=
        scale;
    }
  }


  if (
    !track.velocity_initialized)
  {
    track.velocity =
      measured_velocity;

    track.velocity_initialized =
      true;

  } else {

    track.velocity.x =
      velocity_smoothing_ *
        measured_velocity.x +
      (
        1.0f -
        velocity_smoothing_
      ) *
        track.velocity.x;


    track.velocity.y =
      velocity_smoothing_ *
        measured_velocity.y +
      (
        1.0f -
        velocity_smoothing_
      ) *
        track.velocity.y;
  }


  // Real detector observation becomes authoritative.
  track.bbox =
    det_box;

  track.last_observed_center =
    det_center;

  track.confidence =
    detection.confidence;

  track.hits++;

  track.hit_streak++;

  track.time_since_update =
    0;


  // Lost tracks recover immediately because they were already confirmed.
  if (
    track.state ==
    TrackState::Lost)
  {
    track.state =
      TrackState::Confirmed;

  } else if (
    track.state ==
      TrackState::Tentative &&
    track.hit_streak >=
      min_hits_)
  {
    track.state =
      TrackState::Confirmed;
  }


  if (timestamp_valid) {

    if (
      track.last_observation_time_sec <= 0.0 ||
      timestamp_sec >
        track.last_observation_time_sec)
    {
      track.last_observation_time_sec =
        timestamp_sec;
    }
  }


  if (refresh_appearance) {

    maybeRefreshAppearance(
      track,
      det_box,
      frame
    );

  } else {

    // A Stage-2 appearance recovery should not immediately use the
    // same recovery crop to modify the gallery that authorized it.
    //
    // Wait for future normal Stage-1 matches before refreshing again.
    auto appearance_it =
      appearance_states_.find(
        track.id
      );


    if (
      appearance_it !=
      appearance_states_.end())
    {
      appearance_it->second
        .successful_matches_since_refresh =
          0;
    }
  }
}


// -----------------------------------------------------------------------------
// Main tracker update.
// -----------------------------------------------------------------------------

std::vector<Track> TrackerEngine::update(
  const std::vector<
    opl_interfaces::msg::TrackedHuman>& detections,
  const cv::Mat& frame,
  double timestamp_sec)
{
  const bool timestamp_valid =
    std::isfinite(
      timestamp_sec
    ) &&
    timestamp_sec > 0.0;


  // ===========================================================================
  // Step 1:
  // Global camera motion.
  // ===========================================================================

  const cv::Mat motion =
    estimateCameraMotion(
      frame
    );


  const bool valid_motion =
    !motion.empty() &&
    motion.rows == 2 &&
    motion.cols == 3 &&
    cv::checkRange(
      motion
    );


  auto transformPoint =
    [&](const cv::Point2f& p)
    -> cv::Point2f
  {
    if (!valid_motion) {
      return p;
    }


    const double a =
      motion.at<double>(
        0,
        0
      );

    const double b =
      motion.at<double>(
        0,
        1
      );

    const double tx =
      motion.at<double>(
        0,
        2
      );

    const double c =
      motion.at<double>(
        1,
        0
      );

    const double d =
      motion.at<double>(
        1,
        1
      );

    const double ty =
      motion.at<double>(
        1,
        2
      );


    return cv::Point2f(
      static_cast<float>(
        a * p.x +
        b * p.y +
        tx
      ),

      static_cast<float>(
        c * p.x +
        d * p.y +
        ty
      )
    );
  };


  // ===========================================================================
  // Step 2:
  // GMC + timestamp-aware person-motion prediction.
  // ===========================================================================

  for (
    auto& track :
    active_tracks_)
  {
    track.age++;

    track.time_since_update++;


    double prediction_dt =
      nominal_dt_sec_;


    if (
      timestamp_valid &&
      track.last_prediction_time_sec > 0.0)
    {
      const double measured_dt =
        timestamp_sec -
        track.last_prediction_time_sec;


      if (
        measured_dt >
        min_valid_dt_sec_)
      {
        prediction_dt =
          std::min(
            measured_dt,
            max_prediction_dt_sec_
          );

      } else {

        prediction_dt =
          0.0;
      }

    } else if (
      timestamp_valid &&
      track.last_prediction_time_sec <= 0.0)
    {
      prediction_dt =
        nominal_dt_sec_;
    }


    cv::Point2f center(
      track.bbox.x +
        track.bbox.width *
        0.5f,

      track.bbox.y +
        track.bbox.height *
        0.5f
    );


    center =
      transformPoint(
        center
      );


    // Keep the last real observation in the current camera coordinate system.
    track.last_observed_center =
      transformPoint(
        track.last_observed_center
      );


    if (
      track.velocity_initialized &&
      prediction_dt > 0.0)
    {
      center.x +=
        track.velocity.x *
        static_cast<float>(
          prediction_dt
        );


      center.y +=
        track.velocity.y *
        static_cast<float>(
          prediction_dt
        );
    }


    track.bbox.x =
      center.x -
      track.bbox.width *
      0.5f;


    track.bbox.y =
      center.y -
      track.bbox.height *
      0.5f;


    if (timestamp_valid) {

      if (
        track.last_prediction_time_sec <= 0.0 ||
        timestamp_sec >
          track.last_prediction_time_sec)
      {
        track.last_prediction_time_sec =
          timestamp_sec;
      }
    }
  }


  // ===========================================================================
  // Step 3:
  // Primary IoU association matrix.
  // ===========================================================================

  const std::size_t track_count =
    active_tracks_.size();

  const std::size_t detection_count =
    detections.size();


  constexpr float INVALID_ASSOCIATION_COST =
    1000000.0f;


  std::vector<std::vector<float>>
    iou_matrix(
      track_count,
      std::vector<float>(
        detection_count,
        0.0f
      )
    );


  std::vector<std::vector<float>>
    primary_cost_matrix(
      track_count,
      std::vector<float>(
        detection_count,
        INVALID_ASSOCIATION_COST
      )
    );


  for (
    std::size_t i = 0;
    i < track_count;
    ++i)
  {
    for (
      std::size_t j = 0;
      j < detection_count;
      ++j)
    {
      const cv::Rect2f det_box =
        detectionBox(
          detections[j]
        );


      const float iou =
        computeIoU(
          active_tracks_[i].bbox,
          det_box
        );


      iou_matrix[i][j] =
        iou;


      if (
        iou >=
        iou_threshold_)
      {
        primary_cost_matrix[i][j] =
          1.0f -
          iou;
      }
    }
  }


  const std::vector<int>
    primary_assignment =
      solveHungarian(
        primary_cost_matrix
      );


  std::vector<bool>
    track_matched(
      track_count,
      false
    );


  std::vector<bool>
    det_matched(
      detection_count,
      false
    );


  // ===========================================================================
  // Step 4:
  // Apply Stage-1 IoU/Hungarian matches.
  // ===========================================================================

  for (
    std::size_t track_idx = 0;
    track_idx < track_count;
    ++track_idx)
  {
    if (
      track_idx >=
      primary_assignment.size())
    {
      continue;
    }


    const int assigned_detection =
      primary_assignment[
        track_idx
      ];


    if (
      assigned_detection < 0)
    {
      continue;
    }


    const std::size_t det_idx =
      static_cast<std::size_t>(
        assigned_detection
      );


    if (
      det_idx >=
      detection_count)
    {
      continue;
    }


    if (
      primary_cost_matrix[
        track_idx
      ][
        det_idx
      ] >=
      INVALID_ASSOCIATION_COST *
        0.5f)
    {
      continue;
    }


    if (
      iou_matrix[
        track_idx
      ][
        det_idx
      ] <
      iou_threshold_)
    {
      continue;
    }


    if (
      track_matched[
        track_idx
      ] ||
      det_matched[
        det_idx
      ])
    {
      continue;
    }


    applyDetectionToTrack(
      active_tracks_[
        track_idx
      ],
      detections[
        det_idx
      ],
      frame,
      timestamp_sec,

      // Normal geometric match:
      // appearance may be refreshed at low cadence.
      true
    );


    track_matched[
      track_idx
    ] =
      true;


    det_matched[
      det_idx
    ] =
      true;
  }


  // ===========================================================================
  // Step 5:
  // Secondary appearance-assisted recovery.
  //
  // Only:
  //   - previously confirmed / currently lost tracks
  //   - with an existing appearance gallery
  //   - still within appearance_reid_max_age_
  //
  // and:
  //   - detections left unmatched by Stage 1
  //
  // participate.
  // ===========================================================================

  if (
    appearanceReidReady() &&
    !frame.empty())
  {
    std::vector<std::size_t>
      appearance_track_indices;


    std::vector<std::size_t>
      unmatched_detection_indices;


    // ------------------------------------------------------------------------
    // Eligible old tracks.
    // ------------------------------------------------------------------------

    for (
      std::size_t i = 0;
      i < track_count;
      ++i)
    {
      if (track_matched[i]) {
        continue;
      }


      const Track& track =
        active_tracks_[i];


      // A currently Confirmed track that failed Stage 1 is also eligible:
      // Stage 2 may recover it without ever creating a real missed frame.
      const bool established_track =
        track.state ==
          TrackState::Confirmed ||
        track.state ==
          TrackState::Lost;


      if (!established_track) {
        continue;
      }


      if (
        track.time_since_update >
        appearance_reid_max_age_)
      {
        continue;
      }


      auto appearance_it =
        appearance_states_.find(
          track.id
        );


      if (
        appearance_it ==
          appearance_states_.end() ||
        appearance_it->second
          .body_gallery.empty())
      {
        continue;
      }


      appearance_track_indices.push_back(
        i
      );
    }


    // ------------------------------------------------------------------------
    // Detections Stage 1 failed to consume.
    // ------------------------------------------------------------------------

    for (
      std::size_t j = 0;
      j < detection_count;
      ++j)
    {
      if (!det_matched[j]) {

        unmatched_detection_indices.push_back(
          j
        );
      }
    }


    if (
      !appearance_track_indices.empty() &&
      !unmatched_detection_indices.empty())
    {
      // ----------------------------------------------------------------------
      // Extract each unmatched detection only ONCE.
      // ----------------------------------------------------------------------

      std::vector<std::vector<float>>
        unmatched_detection_embeddings(
          unmatched_detection_indices.size()
        );


      for (
        std::size_t local_det_idx = 0;
        local_det_idx <
          unmatched_detection_indices.size();
        ++local_det_idx)
      {
        const std::size_t global_det_idx =
          unmatched_detection_indices[
            local_det_idx
          ];


        unmatched_detection_embeddings[
          local_det_idx
        ] =
          extractAppearance(
            frame,
            detectionBox(
              detections[
                global_det_idx
              ]
            )
          );
      }


      // ----------------------------------------------------------------------
      // Build Stage-2 appearance cost matrix.
      // ----------------------------------------------------------------------

      std::vector<std::vector<float>>
        appearance_cost_matrix(
          appearance_track_indices.size(),
          std::vector<float>(
            unmatched_detection_indices.size(),
            INVALID_ASSOCIATION_COST
          )
        );


      std::vector<std::vector<float>>
        appearance_similarity_matrix(
          appearance_track_indices.size(),
          std::vector<float>(
            unmatched_detection_indices.size(),
            -1.0f
          )
        );


      for (
        std::size_t local_track_idx = 0;
        local_track_idx <
          appearance_track_indices.size();
        ++local_track_idx)
      {
        const std::size_t global_track_idx =
          appearance_track_indices[
            local_track_idx
          ];


        const Track& track =
          active_tracks_[
            global_track_idx
          ];


        const auto appearance_it =
          appearance_states_.find(
            track.id
          );


        if (
          appearance_it ==
          appearance_states_.end())
        {
          continue;
        }


        for (
          std::size_t local_det_idx = 0;
          local_det_idx <
            unmatched_detection_indices.size();
          ++local_det_idx)
        {
          const std::size_t global_det_idx =
            unmatched_detection_indices[
              local_det_idx
            ];


          const auto& embedding =
            unmatched_detection_embeddings[
              local_det_idx
            ];


          if (embedding.empty()) {
            continue;
          }


          const cv::Rect2f det_box =
            detectionBox(
              detections[
                global_det_idx
              ]
            );


          // Geometry is a gate, not the appearance score itself.
          if (
            !passesAppearanceSpatialGate(
              track,
              det_box,
              frame,
              timestamp_sec
            ))
          {
            continue;
          }


          const float similarity =
            getMaxAppearanceSimilarity(
              embedding,
              appearance_it->second
                .body_gallery
            );


          appearance_similarity_matrix[
            local_track_idx
          ][
            local_det_idx
          ] =
            similarity;


          if (
            similarity >=
            appearance_match_threshold_)
          {
            appearance_cost_matrix[
              local_track_idx
            ][
              local_det_idx
            ] =
              1.0f -
              similarity;
          }
        }
      }


      const std::vector<int>
        appearance_assignment =
          solveHungarian(
            appearance_cost_matrix
          );


      // ----------------------------------------------------------------------
      // Apply Stage-2 matches.
      // ----------------------------------------------------------------------

      for (
        std::size_t local_track_idx = 0;
        local_track_idx <
          appearance_track_indices.size();
        ++local_track_idx)
      {
        if (
          local_track_idx >=
          appearance_assignment.size())
        {
          continue;
        }


        const int assigned_local_detection =
          appearance_assignment[
            local_track_idx
          ];


        if (
          assigned_local_detection < 0)
        {
          continue;
        }


        const std::size_t local_det_idx =
          static_cast<std::size_t>(
            assigned_local_detection
          );


        if (
          local_det_idx >=
          unmatched_detection_indices.size())
        {
          continue;
        }


        if (
          appearance_cost_matrix[
            local_track_idx
          ][
            local_det_idx
          ] >=
          INVALID_ASSOCIATION_COST *
            0.5f)
        {
          continue;
        }


        const float similarity =
          appearance_similarity_matrix[
            local_track_idx
          ][
            local_det_idx
          ];


        if (
          similarity <
          appearance_match_threshold_)
        {
          continue;
        }


        const std::size_t global_track_idx =
          appearance_track_indices[
            local_track_idx
          ];


        const std::size_t global_det_idx =
          unmatched_detection_indices[
            local_det_idx
          ];


        // Defensive one-to-one check.
        if (
          track_matched[
            global_track_idx
          ] ||
          det_matched[
            global_det_idx
          ])
        {
          continue;
        }


        applyDetectionToTrack(
          active_tracks_[
            global_track_idx
          ],
          detections[
            global_det_idx
          ],
          frame,
          timestamp_sec,

          // IMPORTANT:
          // Do not immediately learn the embedding that was used
          // to authorize the recovery.
          false
        );


        track_matched[
          global_track_idx
        ] =
          true;


        det_matched[
          global_det_idx
        ] =
          true;
      }
    }
  }


  // ===========================================================================
  // Step 6:
  // Tracks unmatched by BOTH association stages become genuinely missed.
  // ===========================================================================

  for (
    std::size_t i = 0;
    i < track_count;
    ++i)
  {
    if (track_matched[i]) {
      continue;
    }


    Track& track =
      active_tracks_[i];


    // Actual missed tracker update:
    // consecutive confirmation streak ends.
    track.hit_streak =
      0;


    if (
      track.state ==
      TrackState::Confirmed)
    {
      track.state =
        TrackState::Lost;
    }


    if (
      track.time_since_update >
      velocity_decay_after_)
    {
      track.velocity.x *=
        velocity_decay_;

      track.velocity.y *=
        velocity_decay_;
    }
  }


  // ===========================================================================
  // Step 7:
  // Only detections unmatched by BOTH stages create new raw IDs.
  // ===========================================================================

  for (
    std::size_t j = 0;
    j < detection_count;
    ++j)
  {
    if (det_matched[j]) {
      continue;
    }


    Track new_track;


    new_track.id =
      next_id_++;


    new_track.bbox =
      detectionBox(
        detections[j]
      );


    new_track.confidence =
      detections[j]
        .confidence;


    new_track.hits =
      1;


    new_track.hit_streak =
      1;


    new_track.age =
      1;


    new_track.time_since_update =
      0;


    if (
      min_hits_ <= 1)
    {
      new_track.state =
        TrackState::Confirmed;

    } else {

      new_track.state =
        TrackState::Tentative;
    }


    new_track.velocity =
      cv::Point2f(
        0.0f,
        0.0f
      );


    new_track.velocity_initialized =
      false;


    new_track.last_observed_center =
      cv::Point2f(
        new_track.bbox.x +
          new_track.bbox.width *
          0.5f,

        new_track.bbox.y +
          new_track.bbox.height *
          0.5f
      );


    if (timestamp_valid) {

      new_track.last_prediction_time_sec =
        timestamp_sec;


      new_track.last_observation_time_sec =
        timestamp_sec;
    }


    // If min_hits == 1, seed appearance immediately.
    if (
      new_track.state ==
      TrackState::Confirmed)
    {
      maybeRefreshAppearance(
        new_track,
        new_track.bbox,
        frame
      );
    }


    active_tracks_.push_back(
      new_track
    );
  }


  // ===========================================================================
  // Step 8:
  // Lifecycle / publishing.
  // ===========================================================================

  std::vector<Track>
    confirmed_tracks;


  auto it =
    active_tracks_.begin();


  while (
    it !=
    active_tracks_.end())
  {
    if (
      it->time_since_update >
      max_age_)
    {
      // Appearance state has exactly the same lifetime as its raw track.
      appearance_states_.erase(
        it->id
      );


      it =
        active_tracks_.erase(
          it
        );


      continue;
    }


    const bool publishable_state =
      it->state ==
        TrackState::Confirmed ||
      it->state ==
        TrackState::Lost;


    if (
      publishable_state &&
      it->time_since_update <=
        max_publish_age_)
    {
      confirmed_tracks.push_back(
        *it
      );
    }


    ++it;
  }


  return confirmed_tracks;
}


}  // namespace opl_human_vision