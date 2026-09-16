#include "opl_human_vision/face_engine.hpp"
#include "opl_human_vision/scoped_timing.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace opl_human_vision {

FaceEngine::FaceEngine(
  const std::string& scrfd_path,
  const std::string& adaface_path,
  const std::string& osnet_path,
  float face_match_threshold,
  float body_match_threshold,
  int min_confirm_frames,
  float min_confirm_duration_sec,
  double session_reid_timeout_sec,
  double spatial_gate_max_age_sec,
  float spatial_gate_base_m,
  float spatial_gate_max_speed_mps,
  float depth_gate_base_m)
: face_match_threshold_(face_match_threshold),
  body_match_threshold_(body_match_threshold),
  min_confirm_frames_(min_confirm_frames),
  min_confirm_duration_sec_(min_confirm_duration_sec),
  session_reid_timeout_sec_(session_reid_timeout_sec),
  reid_spatial_gate_max_age_sec_(spatial_gate_max_age_sec),
  reid_spatial_gate_base_m_(spatial_gate_base_m),
  reid_spatial_gate_max_speed_mps_(spatial_gate_max_speed_mps),
  reid_depth_gate_base_m_(depth_gate_base_m)
{
  scrfd_ = std::make_unique<ScrfdDetector>(
    scrfd_path
  );

  adaface_ = std::make_unique<AdaFaceExtractor>(
    adaface_path
  );

  osnet_ = std::make_unique<OsnetExtractor>(
    osnet_path
  );

  last_cleanup_time_ =
    std::chrono::steady_clock::now();
}

bool FaceEngine::loadModels() {
  bool ok = scrfd_->init() && adaface_->init() && osnet_->init();
  if (ok) std::cout << "[FaceEngine] All TensorRT modules initialized successfully." << std::endl;
  return ok;
}

void FaceEngine::cleanupStaleRawTracks(
  const std::set<uint64_t>& active_raw_ids)
{
  const auto now =
    std::chrono::steady_clock::now();


  // --------------------------------------------------------------------------
  // Refresh session identities that still have an active raw tracker.
  // --------------------------------------------------------------------------

  for (const auto& mapping :
       active_raw_to_session_map_)
  {
    const uint64_t raw_id =
      mapping.first;

    const uint64_t session_id =
      mapping.second;


    if (
      active_raw_ids.find(raw_id) ==
      active_raw_ids.end())
    {
      continue;
    }


    auto session_it =
      std::find_if(
        session_gallery_.begin(),
        session_gallery_.end(),
        [session_id](
          const SessionIdentity& item)
        {
          return
            item.session_id ==
            session_id;
        }
      );


    if (
      session_it !=
      session_gallery_.end())
    {
      session_it->last_seen =
        now;
    }
  }


  // --------------------------------------------------------------------------
  // Remove inactive raw -> permanent mappings.
  // --------------------------------------------------------------------------

  for (
    auto it =
      active_raw_to_persistent_map_.begin();

    it !=
      active_raw_to_persistent_map_.end();)
  {
    if (
      active_raw_ids.find(it->first) ==
      active_raw_ids.end())
    {
      it =
        active_raw_to_persistent_map_.erase(
          it
        );

    } else {

      ++it;
    }
  }


  // --------------------------------------------------------------------------
  // Remove inactive raw -> session mappings.
  //
  // The SessionIdentity itself is NOT deleted here.
  // It remains available for ReID until its timeout expires.
  // --------------------------------------------------------------------------

  for (
    auto it =
      active_raw_to_session_map_.begin();

    it !=
      active_raw_to_session_map_.end();)
  {
    if (
      active_raw_ids.find(it->first) ==
      active_raw_ids.end())
    {
      it =
        active_raw_to_session_map_.erase(
          it
        );

    } else {

      ++it;
    }
  }


  // Remove unused frame counters.
  for (
    auto it =
      track_frame_counters_.begin();

    it !=
      track_frame_counters_.end();)
  {
    if (
      active_raw_ids.find(it->first) ==
      active_raw_ids.end())
    {
      it =
        track_frame_counters_.erase(
          it
        );

    } else {

      ++it;
    }
  }


  cleanupStaleSessionIdentities();
}

void FaceEngine::cleanupStaleSessionIdentities()
{
  const auto now =
    std::chrono::steady_clock::now();


  session_gallery_.erase(

    std::remove_if(
      session_gallery_.begin(),
      session_gallery_.end(),

      [&](const SessionIdentity& item)
      {
        const double age_sec =
          std::chrono::duration<double>(
            now - item.last_seen
          ).count();


        return
          age_sec >
          session_reid_timeout_sec_;
      }
    ),

    session_gallery_.end()
  );
}


void FaceEngine::removeSessionIdentity(
  uint64_t session_id)
{
  session_gallery_.erase(

    std::remove_if(
      session_gallery_.begin(),
      session_gallery_.end(),

      [session_id](
        const SessionIdentity& item)
      {
        return
          item.session_id ==
          session_id;
      }
    ),

    session_gallery_.end()
  );


  for (
    auto it =
      active_raw_to_session_map_.begin();

    it !=
      active_raw_to_session_map_.end();)
  {
    if (
      it->second ==
      session_id)
    {
      it =
        active_raw_to_session_map_.erase(
          it
        );

    } else {

      ++it;
    }
  }
}

void FaceEngine::cleanupStalePendingTracks() {
  auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration<double>(now - last_cleanup_time_).count() < 1.0) return;

  for (auto it = pending_enrollments_.begin(); it != pending_enrollments_.end(); ) {
    if (std::chrono::duration<double>(now - it->second.last_seen).count() > 3.0) {
      authorized_enrollments_.erase(it->first);
      custom_names_request_.erase(it->first);
      it = pending_enrollments_.erase(it);
    } else {
      ++it;
    }
  }
  last_cleanup_time_ = now;
}

float FaceEngine::getMaxCosineSimilarity(const std::vector<float>& query, const std::vector<std::vector<float>>& gallery) {
  if (query.empty() || gallery.empty()) return 0.0f;
  float max_sim = 0.0f;
  for (const auto& temp : gallery) {
    if (temp.size() != query.size()) continue;
    float dot = 0.0f;
    for (size_t i = 0; i < query.size(); ++i) dot += query[i] * temp[i];
    if (dot > max_sim) max_sim = dot;
  }
  return max_sim;
}

bool FaceEngine::isValid3DPosition(
  const geometry_msgs::msg::Point& p) const
{
  return
    std::isfinite(p.x) &&
    std::isfinite(p.y) &&
    std::isfinite(p.z) &&
    p.z > 0.20 &&
    p.z < 10.0;
}


bool FaceEngine::passesSpatialGate(
  const geometry_msgs::msg::Point& current,
  const geometry_msgs::msg::Point& previous,
  bool has_previous,
  const std::chrono::steady_clock::time_point& previous_time,
  const std::chrono::steady_clock::time_point& now) const
{
  // If depth is missing/unreliable, do not reject an otherwise
  // plausible appearance match.
  if (!isValid3DPosition(current) || !has_previous) {
    return true;
  }

  if (!isValid3DPosition(previous)) {
    return true;
  }


  double dt =
    std::chrono::duration<double>(
      now - previous_time
    ).count();


  if (dt < 0.0) {
    return true;
  }


  // Old position information should not constrain somebody indefinitely.
  if (dt > reid_spatial_gate_max_age_sec_) {
    return true;
  }


  const double dx =
    current.x - previous.x;

  const double dy =
    current.y - previous.y;

  const double dz =
    current.z - previous.z;


  const double distance_3d =
    std::sqrt(
      dx * dx +
      dy * dy +
      dz * dz
    );


  const double depth_difference =
    std::abs(dz);


  // Allowed displacement grows with elapsed time.
  //
  // Example:
  // dt = 0.5 sec
  //
  // allowed =
  //   0.75 + 2.5 * 0.5
  //   = 2.0 metres
  //
  // This allows rapid human motion while rejecting impossible jumps.
  const double allowed_distance =
    reid_spatial_gate_base_m_ +
    reid_spatial_gate_max_speed_mps_ *
    dt;


  const double allowed_depth_change =
    reid_depth_gate_base_m_ +
    reid_spatial_gate_max_speed_mps_ *
    dt;


  if (distance_3d > allowed_distance) {
    return false;
  }


  if (depth_difference > allowed_depth_change) {
    return false;
  }


  return true;
}


void FaceEngine::updateRecentPosition(
  const geometry_msgs::msg::Point& current,
  geometry_msgs::msg::Point& stored,
  bool& has_stored,
  std::chrono::steady_clock::time_point& stored_time,
  const std::chrono::steady_clock::time_point& now)
{
  if (!isValid3DPosition(current)) {
    return;
  }


  stored = current;
  has_stored = true;
  stored_time = now;
}

void FaceEngine::addTemplateIfDiverse(std::vector<std::vector<float>>& gallery, const std::vector<float>& new_emb, size_t max_size, float thresh) {
  if (new_emb.empty()) return;
  if (gallery.empty()) { gallery.push_back(new_emb); return; }
  if (getMaxCosineSimilarity(new_emb, gallery) < thresh) {
    if (gallery.size() >= max_size) gallery.erase(gallery.begin());
    gallery.push_back(new_emb);
  }
}

std::vector<float> FaceEngine::computeAverageEmbedding(const std::vector<std::vector<float>>& embeddings) {
  if (embeddings.empty()) return {};
  size_t dim = embeddings[0].size();
  std::vector<float> avg(dim, 0.0f);
  for (const auto& emb : embeddings) {
    if (emb.size() != dim) continue;
    for (size_t i = 0; i < dim; ++i) avg[i] += emb[i];
  }

  float norm = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    avg[i] /= static_cast<float>(embeddings.size());
    norm += avg[i] * avg[i];
  }
  norm = std::sqrt(norm);
  if (norm < 1e-6f) return std::vector<float>(dim, 0.0f);

  for (size_t i = 0; i < dim; ++i) avg[i] /= norm;
  return avg;
}

void FaceEngine::updateIdentityPosition(
  uint64_t identity_id,
  const geometry_msgs::msg::Point& position_3d)
{
  if (!isValid3DPosition(position_3d)) {
    return;
  }

  const auto now =
    std::chrono::steady_clock::now();


  // Registered identity.
  for (auto& item : reid_gallery_) {

    if (item.persistent_id == identity_id) {

      updateRecentPosition(
        position_3d,
        item.last_position_3d,
        item.has_position_3d,
        item.last_position_time,
        now
      );

      return;
    }
  }


  // Anonymous/session identity.
  for (auto& item : session_gallery_) {

    if (item.session_id == identity_id) {

      item.last_seen = now;

      updateRecentPosition(
        position_3d,
        item.last_position_3d,
        item.has_position_3d,
        item.last_position_time,
        now
      );

      return;
    }
  }
}

FaceResult FaceEngine::processFace(
  const cv::Mat& human_crop,
  uint64_t raw_track_id,
  const std::set<uint64_t>& occupied_ids,
  const geometry_msgs::msg::Point& position_3d),
  bool force_deep_reid)
{
  ScopedTiming timing(
    "FaceEngine.processFace"
  );


  cleanupStalePendingTracks();
  cleanupStaleSessionIdentities();


  FaceResult result;


  if (
    human_crop.empty() ||
    human_crop.cols < 15 ||
    human_crop.rows < 15)
  {
    return result;
  }


  const auto now =
    std::chrono::steady_clock::now();


  // --------------------------------------------------------------------------
  // Step 1: Detect face + landmarks.
  // --------------------------------------------------------------------------

  std::vector<cv::Point2f> landmarks;

  float face_score = 0.0f;


  const bool face_detected =
    scrfd_->detect(
      human_crop,
      landmarks,
      face_score
    );


  if (face_detected) {
    result.landmarks =
      landmarks;
  }


  // --------------------------------------------------------------------------
  // Step 2: Face-quality gate.
  // --------------------------------------------------------------------------

  bool valid_face_quality =
    false;


  if (
    face_detected &&
    landmarks.size() == 5)
  {
    const float eye_dist =
      cv::norm(
        landmarks[0] -
        landmarks[1]
      );


    const bool eye_dist_ok =
      eye_dist >= 9.0f;


    bool landmarks_inside =
      true;


    constexpr float margin =
      4.0f;


    for (const auto& lm : landmarks) {

      if (
        lm.x < margin ||
        lm.x >
          human_crop.cols - margin ||
        lm.y < margin ||
        lm.y >
          human_crop.rows - margin)
      {
        landmarks_inside =
          false;

        break;
      }
    }


    valid_face_quality =
      eye_dist_ok &&
      landmarks_inside;
  }


  // --------------------------------------------------------------------------
  // Step 3: Inspect existing raw-track mappings.
  // --------------------------------------------------------------------------

  bool enrollment_authorized =
    false;

  bool is_persistent_track =
    false;

  bool is_session_track =
    false;


  uint64_t cached_persistent_id =
    0;

  uint64_t cached_session_id =
    0;


  if (raw_track_id != 0) {

    track_frame_counters_[
      raw_track_id
    ]++;


    enrollment_authorized =
      authorized_enrollments_.find(
        raw_track_id
      ) !=
      authorized_enrollments_.end();


    auto persistent_it =
      active_raw_to_persistent_map_.find(
        raw_track_id
      );


    if (
      persistent_it !=
      active_raw_to_persistent_map_.end())
    {
      is_persistent_track =
        true;

      cached_persistent_id =
        persistent_it->second;
    }


    auto session_it =
      active_raw_to_session_map_.find(
        raw_track_id
      );


    if (
      session_it !=
      active_raw_to_session_map_.end())
    {
      is_session_track =
        true;

      cached_session_id =
        session_it->second;
    }
  }

    if (
    is_persistent_track &&
    occupied_ids.find(cached_persistent_id) !=
      occupied_ids.end())
  {
    active_raw_to_persistent_map_.erase(
      raw_track_id
    );

    is_persistent_track =
      false;

    cached_persistent_id =
      0;
  }

  if (
    is_session_track &&
    occupied_ids.find(cached_session_id) !=
      occupied_ids.end())
  {
    active_raw_to_session_map_.erase(
      raw_track_id
    );

    is_session_track =
      false;

    cached_session_id =
      0;
  }

  const bool has_cached_identity =
    is_persistent_track ||
    is_session_track;


  // Enrollment must force feature extraction.
  const bool run_deep_reid =
    force_deep_reid ||
    enrollment_authorized ||
    !has_cached_identity ||
    (
      raw_track_id != 0 &&
      track_frame_counters_[
        raw_track_id
      ] % 6 == 0
    );


  // --------------------------------------------------------------------------
  // Fast path: permanent registered identity.
  // --------------------------------------------------------------------------

  if (
    is_persistent_track &&
    !run_deep_reid)
  {
    auto gallery_it =
      std::find_if(
        reid_gallery_.begin(),
        reid_gallery_.end(),

        [cached_persistent_id](
          const PersistentIdentity& item)
        {
          return
            item.persistent_id ==
            cached_persistent_id;
        }
      );


    if (
      gallery_it !=
      reid_gallery_.end())
    {
      result.persistent_id =
        cached_persistent_id;

      result.name =
        gallery_it->name;

      result.match_score =
        1.0f;

      return result;
    }
  }


  // --------------------------------------------------------------------------
  // Fast path: anonymous session identity.
  //
  // Do NOT use this path when enrollment is authorized.
  // Enrollment needs real feature extraction.
  // --------------------------------------------------------------------------

  if (
    is_session_track &&
    !run_deep_reid &&
    !enrollment_authorized)
  {
    auto session_it =
      std::find_if(
        session_gallery_.begin(),
        session_gallery_.end(),

        [cached_session_id](
          const SessionIdentity& item)
        {
          return
            item.session_id ==
            cached_session_id;
        }
      );


    if (
      session_it !=
      session_gallery_.end())
    {
      session_it->last_seen =
        now;


      result.persistent_id =
        cached_session_id;

      result.name =
        "unknown";

      result.match_score =
        1.0f;

      return result;
    }
  }


  // --------------------------------------------------------------------------
  // Step 4: Deep feature extraction.
  // --------------------------------------------------------------------------

  result.body_embedding =
    osnet_->extract(
      human_crop
    );


  if (valid_face_quality) {

    cv::Mat aligned_face =
      adaface_->alignFace(
        human_crop,
        landmarks
      );


    result.face_embedding =
      adaface_->extract(
        aligned_face
      );
  }

  // --------------------------------------------------------------------------
  // Step 5: Validate an existing registered mapping.
  // --------------------------------------------------------------------------
  if (is_persistent_track) {

    auto gallery_it =
      std::find_if(
        reid_gallery_.begin(),
        reid_gallery_.end(),

        [cached_persistent_id](
          const PersistentIdentity& item)
        {
          return
            item.persistent_id ==
            cached_persistent_id;
        }
      );


    if (
      gallery_it !=
      reid_gallery_.end())
    {
      const float face_sim =
        getMaxCosineSimilarity(
          result.face_embedding,
          gallery_it->face_gallery
        );


      const float body_sim =
        getMaxCosineSimilarity(
          result.body_embedding,
          gallery_it->body_gallery
        );


      bool has_comparable_embedding =
        false;

      bool valid_match =
        false;


      if (
        !result.face_embedding.empty() &&
        gallery_it->has_face_embedding())
      {
        has_comparable_embedding =
          true;

        valid_match =
          face_sim >=
          face_match_threshold_ *
            0.75f;

      } else if (
        !result.body_embedding.empty() &&
        gallery_it->has_body_embedding())
      {
        has_comparable_embedding =
          true;

        // Preserve your existing persistent-body
        // validation multiplier.
        valid_match =
          body_sim >=
          body_match_threshold_ *
            0.55f;
      }


      // No comparable feature means validation is inconclusive,
      // not proof that the identity is wrong.
      //
      // Keep the mapping and retry on a future validation.
      if (!has_comparable_embedding) {

        result.persistent_id =
          cached_persistent_id;

        result.name =
          gallery_it->name;

        result.match_score =
          0.0f;


        updateRecentPosition(
          position_3d,
          gallery_it->last_position_3d,
          gallery_it->has_position_3d,
          gallery_it->last_position_time,
          now
        );


        return result;
      }


      if (valid_match) {

        addTemplateIfDiverse(
          gallery_it->body_gallery,
          result.body_embedding,
          10,
          0.82f
        );


        if (
          !result.face_embedding.empty())
        {
          addTemplateIfDiverse(
            gallery_it->face_gallery,
            result.face_embedding,
            6,
            0.85f
          );
        }


        result.persistent_id =
          cached_persistent_id;

        result.name =
          gallery_it->name;


        // Unlike a pure cache hit, this is a real validation result.
        if (
          !result.face_embedding.empty() &&
          gallery_it->has_face_embedding())
        {
          result.match_score =
            face_sim;

        } else {

          result.match_score =
            body_sim;
        }


        updateRecentPosition(
          position_3d,
          gallery_it->last_position_3d,
          gallery_it->has_position_3d,
          gallery_it->last_position_time,
          now
        );


        return result;

      } else {

        // Actual comparable evidence rejected the cached identity.
        active_raw_to_persistent_map_.erase(
          raw_track_id
        );

        is_persistent_track =
          false;
      }
    }
  }


  // --------------------------------------------------------------------------
  // Step 6: ALWAYS search enrolled identities before anonymous identities.
  //
  // This allows a person initially classified as anonymous to later become
  // recognized when a good face/body view becomes available.
  // --------------------------------------------------------------------------

  int best_registered_idx =
    -1;

  float best_registered_sim =
    -1.0f;


  for (
    size_t i = 0;
    i < reid_gallery_.size();
    ++i)
  {
    const auto& item =
      reid_gallery_[i];


    if (
      occupied_ids.find(
        item.persistent_id
      ) !=
      occupied_ids.end())
    {
      continue;
    }

    if (!passesSpatialGate(
          position_3d,
          item.last_position_3d,
          item.has_position_3d,
          item.last_position_time,
          now))
    {
      continue;
    }


    const float face_sim =
      getMaxCosineSimilarity(
        result.face_embedding,
        item.face_gallery
      );


    const float body_sim =
      getMaxCosineSimilarity(
        result.body_embedding,
        item.body_gallery
      );


    const bool can_use_face =
      !result.face_embedding.empty() &&
      item.has_face_embedding();


    float combined_sim =
      0.0f;


    if (can_use_face) {

      if (
        face_sim >=
        face_match_threshold_)
      {
        combined_sim =
          face_sim;

      } else if (
        item.has_body_embedding())
      {
        combined_sim =
          0.8f * face_sim +
          0.2f * body_sim;

      } else {

        combined_sim =
          face_sim;
      }

    } else {

      combined_sim =
        body_sim;
    }


    const float required_thresh =
      can_use_face
      ? face_match_threshold_
      : body_match_threshold_;


    if (
      combined_sim >=
        required_thresh &&
      combined_sim >
        best_registered_sim)
    {
      best_registered_sim =
        combined_sim;

      best_registered_idx =
        static_cast<int>(i);
    }
  }


  // --------------------------------------------------------------------------
  // Registered identity recovered.
  // --------------------------------------------------------------------------

  if (
    best_registered_idx >= 0)
  {
    auto& matched =
      reid_gallery_[
        best_registered_idx
      ];


    result.persistent_id =
      matched.persistent_id;

    result.name =
      matched.name;

    result.match_score =
      best_registered_sim;
    
    updateRecentPosition(
      position_3d,
      matched.last_position_3d,
      matched.has_position_3d,
      matched.last_position_time,
      now
    );

    addTemplateIfDiverse(
      matched.body_gallery,
      result.body_embedding,
      10,
      0.82f
    );


    if (
      !result.face_embedding.empty())
    {
      addTemplateIfDiverse(
        matched.face_gallery,
        result.face_embedding,
        6,
        0.85f
      );
    }


    if (raw_track_id != 0) {

      // If this raw track previously belonged to an anonymous session
      // identity, that anonymous identity is no longer needed.
      auto old_session_it =
        active_raw_to_session_map_.find(
          raw_track_id
        );


      if (
        old_session_it !=
        active_raw_to_session_map_.end())
      {
        const uint64_t old_session_id =
          old_session_it->second;

        removeSessionIdentity(
          old_session_id
        );
      }


      active_raw_to_persistent_map_[
        raw_track_id
      ] =
        matched.persistent_id;


      pending_enrollments_.erase(
        raw_track_id
      );
    }


    return result;
  }


  // --------------------------------------------------------------------------
  // Step 7: User explicitly requested enrollment.
  //
  // Anonymous session matching is deliberately bypassed here.
  // --------------------------------------------------------------------------

  if (
    raw_track_id != 0 &&
    enrollment_authorized)
  {
    auto& pending =
      pending_enrollments_[
        raw_track_id
      ];


    if (
      pending.frame_count == 0)
    {
      pending.first_seen =
        now;
    }


    pending.last_seen =
      now;

    pending.frame_count++;


    if (
      !result.body_embedding.empty())
    {
      pending.body_embeddings.push_back(
        result.body_embedding
      );


      if (
        pending.body_embeddings.size() >
        20)
      {
        pending.body_embeddings.erase(
          pending.body_embeddings.begin()
        );
      }
    }


    if (
      !result.face_embedding.empty())
    {
      pending.face_embeddings.push_back(
        result.face_embedding
      );


      if (
        pending.face_embeddings.size() >
        20)
      {
        pending.face_embeddings.erase(
          pending.face_embeddings.begin()
        );
      }
    }


    const double elapsed_sec =
      std::chrono::duration<double>(
        now -
        pending.first_seen
      ).count();


    RCLCPP_INFO(
      rclcpp::get_logger(
        "vision_timing"
      ),
      "[timing] enrollment track=%llu "
      "valid_face_frames=%zu/%d "
      "elapsed_sec=%.3f/%.3f",
      static_cast<unsigned long long>(
        raw_track_id
      ),
      pending.face_embeddings.size(),
      min_confirm_frames_,
      elapsed_sec,
      min_confirm_duration_sec_
    );


    if (
      pending.face_embeddings.size() >=
        static_cast<size_t>(
          min_confirm_frames_
        ) &&
      elapsed_sec >=
        min_confirm_duration_sec_)
    {
      PersistentIdentity new_id;

      new_id.persistent_id =
        next_persistent_id_++;

      updateRecentPosition(
        position_3d,
        new_id.last_position_3d,
        new_id.has_position_3d,
        new_id.last_position_time,
        now
      );

      if (
        custom_names_request_.count(
          raw_track_id
        ) &&
        !custom_names_request_[
          raw_track_id
        ].empty())
      {
        new_id.name =
          custom_names_request_[
            raw_track_id
          ];

      } else {

        new_id.name =
          "Person_" +
          std::to_string(
            new_id.persistent_id
          );
      }


      new_id.face_gallery.push_back(
        computeAverageEmbedding(
          pending.face_embeddings
        )
      );


      for (
        const auto& body_emb :
        pending.body_embeddings)
      {
        addTemplateIfDiverse(
          new_id.body_gallery,
          body_emb,
          10,
          0.82f
        );
      }


      reid_gallery_.push_back(
        new_id
      );


      result.persistent_id =
        new_id.persistent_id;

      result.name =
        new_id.name;

      result.match_score =
        1.0f;

      result.just_scanned =
        true;


      // Remove temporary anonymous identity if this raw track had one.
      auto old_session_it =
        active_raw_to_session_map_.find(
          raw_track_id
        );


      if (
        old_session_it !=
        active_raw_to_session_map_.end())
      {
        const uint64_t old_session_id =
          old_session_it->second;

        removeSessionIdentity(
          old_session_id
        );
      }


      active_raw_to_persistent_map_[
        raw_track_id
      ] =
        new_id.persistent_id;


      pending_enrollments_.erase(
        raw_track_id
      );

      authorized_enrollments_.erase(
        raw_track_id
      );

      custom_names_request_.erase(
        raw_track_id
      );


      return result;
    }


    result.persistent_id =
      0;

    const int faces_collected =
      static_cast<int>(
        pending.face_embeddings.size()
      );


    result.name =
      "Scanning (" +
      std::to_string(
        faces_collected
      ) +
      "/" +
      std::to_string(
        min_confirm_frames_
      ) +
      ")";


    return result;
  }

  // --------------------------------------------------------------------------
  // Step 8: Validate current anonymous session mapping.
  // --------------------------------------------------------------------------

  if (is_session_track) {

    auto session_it =
      std::find_if(
        session_gallery_.begin(),
        session_gallery_.end(),

        [cached_session_id](
          const SessionIdentity& item)
        {
          return
            item.session_id ==
            cached_session_id;
        }
      );


    if (
      session_it !=
      session_gallery_.end())
    {
      const float face_sim =
        getMaxCosineSimilarity(
          result.face_embedding,
          session_it->face_gallery
        );


      const float body_sim =
        getMaxCosineSimilarity(
          result.body_embedding,
          session_it->body_gallery
        );


      bool has_comparable_embedding =
        false;

      bool valid_match =
        false;


      if (
        !result.face_embedding.empty() &&
        session_it->has_face_embedding())
      {
        has_comparable_embedding =
          true;

        valid_match =
          face_sim >=
          face_match_threshold_ *
            0.75f;

      } else if (
        !result.body_embedding.empty() &&
        session_it->has_body_embedding())
      {
        has_comparable_embedding =
          true;

        // Preserve your existing session-body multiplier.
        valid_match =
          body_sim >=
          body_match_threshold_ *
            0.75f;
      }


      // No comparable feature = validation inconclusive.
      // Keep this session mapping and retry later.
      if (!has_comparable_embedding) {

        session_it->last_seen =
          now;


        result.persistent_id =
          session_it->session_id;

        result.name =
          "unknown";

        result.match_score =
          0.0f;


        updateRecentPosition(
          position_3d,
          session_it->last_position_3d,
          session_it->has_position_3d,
          session_it->last_position_time,
          now
        );


        return result;
      }


      if (valid_match) {

        addTemplateIfDiverse(
          session_it->body_gallery,
          result.body_embedding,
          8,
          0.82f
        );


        if (
          !result.face_embedding.empty())
        {
          addTemplateIfDiverse(
            session_it->face_gallery,
            result.face_embedding,
            4,
            0.85f
          );
        }


        session_it->last_seen =
          now;


        result.persistent_id =
          session_it->session_id;

        result.name =
          "unknown";


        if (
          !result.face_embedding.empty() &&
          session_it->has_face_embedding())
        {
          result.match_score =
            face_sim;

        } else {

          result.match_score =
            body_sim;
        }


        updateRecentPosition(
          position_3d,
          session_it->last_position_3d,
          session_it->has_position_3d,
          session_it->last_position_time,
          now
        );


        return result;

      } else {

        // Comparable appearance evidence says this is no longer
        // the same anonymous session.
        active_raw_to_session_map_.erase(
          raw_track_id
        );

        is_session_track =
          false;
      }
    }
  }


  // --------------------------------------------------------------------------
  // Step 9: Search anonymous session gallery.
  // --------------------------------------------------------------------------

  int best_session_idx =
    -1;

  float best_session_sim =
    -1.0f;


  for (
    size_t i = 0;
    i < session_gallery_.size();
    ++i)
  {
    const auto& item =
      session_gallery_[i];


    if (
      occupied_ids.find(
        item.session_id
      ) !=
      occupied_ids.end())
    {
      continue;
    }

    if (!passesSpatialGate(
          position_3d,
          item.last_position_3d,
          item.has_position_3d,
          item.last_position_time,
          now))
    {
      continue;
    }

    const float face_sim =
      getMaxCosineSimilarity(
        result.face_embedding,
        item.face_gallery
      );


    const float body_sim =
      getMaxCosineSimilarity(
        result.body_embedding,
        item.body_gallery
      );


    const bool can_use_face =
      !result.face_embedding.empty() &&
      item.has_face_embedding();


    float combined_sim =
      0.0f;


    if (can_use_face) {

      if (
        face_sim >=
        face_match_threshold_)
      {
        combined_sim =
          face_sim;

      } else if (
        item.has_body_embedding())
      {
        combined_sim =
          0.8f * face_sim +
          0.2f * body_sim;

      } else {

        combined_sim =
          face_sim;
      }

    } else {

      combined_sim =
        body_sim;
    }


    const float required_thresh =
      can_use_face
      ? face_match_threshold_
      : body_match_threshold_;


    if (
      combined_sim >=
        required_thresh &&
      combined_sim >
        best_session_sim)
    {
      best_session_sim =
        combined_sim;

      best_session_idx =
        static_cast<int>(i);
    }
  }


  // --------------------------------------------------------------------------
  // Existing anonymous person recovered.
  // --------------------------------------------------------------------------

  if (
    best_session_idx >= 0)
  {
    auto& matched =
      session_gallery_[
        best_session_idx
      ];


    matched.last_seen =
      now;


    addTemplateIfDiverse(
      matched.body_gallery,
      result.body_embedding,
      8,
      0.82f
    );


    if (
      !result.face_embedding.empty())
    {
      addTemplateIfDiverse(
        matched.face_gallery,
        result.face_embedding,
        4,
        0.85f
      );
    }


    result.persistent_id =
      matched.session_id;

    result.name =
      "unknown";

    result.match_score =
      best_session_sim;

    updateRecentPosition(
      position_3d,
      matched.last_position_3d,
      matched.has_position_3d,
      matched.last_position_time,
      now
    );

    if (raw_track_id != 0) {
      active_raw_to_session_map_[
        raw_track_id
      ] = matched.session_id;
    }

    return result;
  }


  // --------------------------------------------------------------------------
  // Step 10: Create a brand-new anonymous session identity.
  // --------------------------------------------------------------------------

  if (
    !result.body_embedding.empty() ||
    !result.face_embedding.empty())
  {
    SessionIdentity new_session;


    new_session.session_id =
      next_session_id_++;

    new_session.last_seen =
      now;
    
    updateRecentPosition(
      position_3d,
      new_session.last_position_3d,
      new_session.has_position_3d,
      new_session.last_position_time,
      now
    );

    if (
      !result.body_embedding.empty())
    {
      new_session.body_gallery.push_back(
        result.body_embedding
      );
    }


    if (
      !result.face_embedding.empty())
    {
      new_session.face_gallery.push_back(
        result.face_embedding
      );
    }


    const uint64_t new_session_id =
      new_session.session_id;


    session_gallery_.push_back(
      std::move(new_session)
    );


    if (raw_track_id != 0) {

      active_raw_to_session_map_[
        raw_track_id
      ] =
        new_session_id;
    }


    result.persistent_id =
      new_session_id;

    result.name =
      "unknown";

    result.match_score =
      1.0f;


    return result;
  }


  // Nothing usable extracted.
  result.persistent_id =
    0;

  result.name =
    "Unregistered";

  result.match_score =
    0.0f;


  return result;
}

} // namespace opl_human_vision
