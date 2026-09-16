#include "opl_human_vision/face_engine.hpp"
#include "opl_human_vision/scoped_timing.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace opl_human_vision {

FaceEngine::FaceEngine(const std::string& scrfd_path,
                       const std::string& adaface_path,
                       const std::string& osnet_path,
                       float face_match_threshold,
                       float body_match_threshold,
                       int min_confirm_frames,
                       float min_confirm_duration_sec)
  : face_match_threshold_(face_match_threshold),
    body_match_threshold_(body_match_threshold),
    min_confirm_frames_(min_confirm_frames),
    min_confirm_duration_sec_(min_confirm_duration_sec) {
  scrfd_ = std::make_unique<ScrfdDetector>(scrfd_path);
  adaface_ = std::make_unique<AdaFaceExtractor>(adaface_path);
  osnet_ = std::make_unique<OsnetExtractor>(osnet_path);
  last_cleanup_time_ = std::chrono::steady_clock::now();
}

bool FaceEngine::loadModels() {
  bool ok = scrfd_->init() && adaface_->init() && osnet_->init();
  if (ok) std::cout << "[FaceEngine] All TensorRT modules initialized successfully." << std::endl;
  return ok;
}

void FaceEngine::cleanupStaleRawTracks(const std::set<uint64_t>& active_raw_ids) {
  for (auto it = active_raw_to_persistent_map_.begin(); it != active_raw_to_persistent_map_.end(); ) {
    if (active_raw_ids.find(it->first) == active_raw_ids.end()) {
      track_frame_counters_.erase(it->first);
      pending_enrollments_.erase(it->first);
      authorized_enrollments_.erase(it->first);
      custom_names_request_.erase(it->first);
      it = active_raw_to_persistent_map_.erase(it);
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

FaceResult FaceEngine::processFace(const cv::Mat& human_crop, uint64_t raw_track_id, const std::set<uint64_t>& occupied_ids) {
  ScopedTiming timing("FaceEngine.processFace");
  cleanupStalePendingTracks();

  FaceResult result;
  if (human_crop.empty() || human_crop.cols < 15 || human_crop.rows < 15) return result;

  // STEP 1: Face Landmark Detection
  std::vector<cv::Point2f> landmarks;
  float face_score = 0.0f;
  bool face_detected = scrfd_->detect(human_crop, landmarks, face_score);
  if (face_detected) result.landmarks = landmarks;

  // STEP 2: Landmark Boundary Quality Checking
  bool valid_face_quality = false;
  if (face_detected && landmarks.size() == 5) {
    float eye_dist = cv::norm(landmarks[0] - landmarks[1]);
    bool eye_dist_ok = (eye_dist >= 9.0f);

    bool landmarks_inside = true;
    float margin = 4.0f;
    for (const auto& lm : landmarks) {
      if (lm.x < margin || lm.x > (human_crop.cols - margin) ||
          lm.y < margin || lm.y > (human_crop.rows - margin)) {
        landmarks_inside = false;
        break;
      }
    }
    if (eye_dist_ok && landmarks_inside) valid_face_quality = true;
  }

  // STEP 3: Active Track Cache Check
  bool is_actively_tracked = false;
  bool run_deep_reid = true;
  std::unordered_map<uint64_t, uint64_t>::iterator active_it;

  if (raw_track_id != 0) {
    track_frame_counters_[raw_track_id]++;
    active_it = active_raw_to_persistent_map_.find(raw_track_id);
    is_actively_tracked = (active_it != active_raw_to_persistent_map_.end());
    run_deep_reid = !is_actively_tracked || (track_frame_counters_[raw_track_id] % 6 == 0);
  }

  if (is_actively_tracked && !run_deep_reid) {
    uint64_t cached_pid = active_it->second;
    for (const auto& item : reid_gallery_) {
      if (item.persistent_id == cached_pid) {
        result.persistent_id = cached_pid;
        result.name = item.name;
        result.match_score = 1.0f;
        return result;
      }
    }
  }

  // STEP 4: Feature Extraction
  result.body_embedding = osnet_->extract(human_crop);

  if (valid_face_quality) {
    cv::Mat aligned_face = adaface_->alignFace(human_crop, landmarks);
    result.face_embedding = adaface_->extract(aligned_face);
  }

  if (is_actively_tracked) {
    uint64_t cached_pid = active_it->second;
    auto gallery_it = std::find_if(reid_gallery_.begin(), reid_gallery_.end(),
      [cached_pid](const PersistentIdentity& item) { return item.persistent_id == cached_pid; });

    if (gallery_it != reid_gallery_.end()) {
      bool valid_active_match = true;

      if (run_deep_reid) {
        float face_sim = getMaxCosineSimilarity(result.face_embedding, gallery_it->face_gallery);
        float body_sim = getMaxCosineSimilarity(result.body_embedding, gallery_it->body_gallery);

        if (!result.face_embedding.empty() && gallery_it->has_face_embedding()) {
          if (face_sim < face_match_threshold_ * 0.75f) valid_active_match = false;
        } else if (!result.body_embedding.empty() && gallery_it->has_body_embedding()) {
          if (body_sim < body_match_threshold_ * 0.55f) valid_active_match = false;
        }
      }

      if (valid_active_match) {
        addTemplateIfDiverse(gallery_it->body_gallery, result.body_embedding, 10, 0.82f);
        if (!result.face_embedding.empty()) {
          addTemplateIfDiverse(gallery_it->face_gallery, result.face_embedding, 6, 0.85f);
        }
        result.persistent_id = cached_pid;
        result.name = gallery_it->name;
        result.match_score = 1.0f;
        return result;
      } else {
        active_raw_to_persistent_map_.erase(active_it);
      }
    }
  }

  // STEP 5: Gallery Matching for Existing Persistent IDs
  int best_match_idx = -1;
  float best_match_sim = -1.0f;

  for (size_t i = 0; i < reid_gallery_.size(); ++i) {
    const auto& item = reid_gallery_[i];
    if (occupied_ids.find(item.persistent_id) != occupied_ids.end()) continue;

    float face_sim = getMaxCosineSimilarity(result.face_embedding, item.face_gallery);
    float body_sim = getMaxCosineSimilarity(result.body_embedding, item.body_gallery);

    float combined_sim = 0.0f;
    if (!result.face_embedding.empty() && item.has_face_embedding()) {
      if (face_sim >= face_match_threshold_) {
        combined_sim = face_sim;
      } else {
        combined_sim = item.has_body_embedding() ? (0.8f * face_sim + 0.2f * body_sim) : face_sim;
      }
    } else {
      combined_sim = body_sim;
    }

    if (combined_sim > best_match_sim) {
      best_match_sim = combined_sim;
      best_match_idx = static_cast<int>(i);
    }
  }

  float required_thresh = (!result.face_embedding.empty()) ? face_match_threshold_ : body_match_threshold_;

  if (best_match_idx >= 0 && best_match_sim >= required_thresh) {
    auto& matched = reid_gallery_[best_match_idx];
    result.persistent_id = matched.persistent_id;
    result.name = matched.name;
    result.match_score = best_match_sim;

    addTemplateIfDiverse(matched.body_gallery, result.body_embedding, 10, 0.82f);
    if (!result.face_embedding.empty()) {
      addTemplateIfDiverse(matched.face_gallery, result.face_embedding, 6, 0.85f);
    }
    if (raw_track_id != 0) pending_enrollments_.erase(raw_track_id);
  }
  else if (raw_track_id != 0) {
    // STEP 6: Candidate Enrollment Logic (Gated by authorization)
    if (authorized_enrollments_.find(raw_track_id) == authorized_enrollments_.end()) {
      result.persistent_id = 0;
      result.name = "Unregistered";
      return result;
    }

    auto& pending = pending_enrollments_[raw_track_id];
    auto now = std::chrono::steady_clock::now();

    if (pending.frame_count == 0) pending.first_seen = now;
    pending.last_seen = now;
    pending.frame_count++;

    if (!result.body_embedding.empty()) {
      pending.body_embeddings.push_back(result.body_embedding);
      if (pending.body_embeddings.size() > 20) pending.body_embeddings.erase(pending.body_embeddings.begin());
    }

    if (!result.face_embedding.empty()) {
      pending.face_embeddings.push_back(result.face_embedding);
      if (pending.face_embeddings.size() > 20) pending.face_embeddings.erase(pending.face_embeddings.begin());
    }

    double elapsed_sec = std::chrono::duration<double>(now - pending.first_seen).count();
    RCLCPP_INFO(rclcpp::get_logger("vision_timing"),
      "[timing] enrollment track=%llu valid_face_frames=%zu/%d elapsed_sec=%.3f/%.3f",
      static_cast<unsigned long long>(raw_track_id), pending.face_embeddings.size(),
      min_confirm_frames_, elapsed_sec, min_confirm_duration_sec_);

    if (pending.face_embeddings.size() >= static_cast<size_t>(min_confirm_frames_) &&
        elapsed_sec >= min_confirm_duration_sec_) {

      PersistentIdentity new_id;
      new_id.persistent_id = next_persistent_id_++;

      if (custom_names_request_.count(raw_track_id) && !custom_names_request_[raw_track_id].empty()) {
        new_id.name = custom_names_request_[raw_track_id];
      } else {
        new_id.name = "Person_" + std::to_string(new_id.persistent_id);
      }

      new_id.face_gallery.push_back(computeAverageEmbedding(pending.face_embeddings));
      for (const auto& body_emb : pending.body_embeddings) {
        addTemplateIfDiverse(new_id.body_gallery, body_emb, 10, 0.82f);
      }

      reid_gallery_.push_back(new_id);

      result.persistent_id = new_id.persistent_id;
      result.name = new_id.name;
      result.match_score = 1.0f;
      result.just_scanned = true;

      pending_enrollments_.erase(raw_track_id);
      authorized_enrollments_.erase(raw_track_id);
      custom_names_request_.erase(raw_track_id);
    } else {
      result.persistent_id = 0;
      int faces_collected = static_cast<int>(pending.face_embeddings.size());
      result.name = "Scanning (" + std::to_string(faces_collected) + "/" + std::to_string(min_confirm_frames_) + ")";
      return result;
    }
  }

  if (raw_track_id != 0 && result.persistent_id > 0) {
    active_raw_to_persistent_map_[raw_track_id] = result.persistent_id;
  }

  return result;
}

} // namespace opl_human_vision
