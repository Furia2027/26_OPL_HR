#include "opl_human_vision/tracker_engine.hpp"
#include <algorithm>
#include <cmath>

namespace opl_human_vision {

TrackerEngine::TrackerEngine(int max_age, int min_hits, float iou_threshold)
: max_age_(max_age), min_hits_(min_hits), iou_threshold_(iou_threshold) {}

cv::Mat TrackerEngine::estimateCameraMotion(const cv::Mat& current_frame) {
  cv::Mat transform = cv::Mat::eye(2, 3, CV_64F);
  if (current_frame.empty()) return transform;

  cv::Mat gray, small_gray;
  if (current_frame.channels() == 3) {
    cv::cvtColor(current_frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = current_frame;
  }

  // Downscale image for fast optical flow processing
  float scale = 1.0f;
  if (gray.cols > 320) {
    scale = 320.0f / static_cast<float>(gray.cols);
    cv::resize(gray, small_gray, cv::Size(), scale, scale, cv::INTER_NEAREST);
  } else {
    small_gray = gray;
  }

  if (prev_frame_gray_.empty()) {
    prev_frame_gray_ = small_gray.clone();
    return transform;
  }

  // Feature Masking: Exclude human bounding boxes so foreground movement doesn't distort GMC
  cv::Mat mask = cv::Mat::ones(small_gray.size(), CV_8UC1) * 255;
  for (const auto& track : active_tracks_) {
    cv::Rect2f scaled_box(
      track.bbox.x * scale, track.bbox.y * scale,
      track.bbox.width * scale, track.bbox.height * scale
    );

    // FIX 1: Explicit float-to-int clamping prevents out-of-bounds heap corruption
    int x1 = std::max(0, static_cast<int>(std::floor(scaled_box.x)));
    int y1 = std::max(0, static_cast<int>(std::floor(scaled_box.y)));
    int x2 = std::min(small_gray.cols, static_cast<int>(std::ceil(scaled_box.x + scaled_box.width)));
    int y2 = std::min(small_gray.rows, static_cast<int>(std::ceil(scaled_box.y + scaled_box.height)));

    if (x2 > x1 && y2 > y1) {
      mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(0);
    }
  }

  std::vector<cv::Point2f> prev_pts, curr_pts;
  // Shi-Tomasi Corner Detection on background elements
  cv::goodFeaturesToTrack(prev_frame_gray_, prev_pts, 200, 0.01, 10, mask);

  if (prev_pts.size() >= 10) {
    std::vector<uchar> status;
    std::vector<float> err;
    // Lucas-Kanade Optical Flow
    cv::calcOpticalFlowPyrLK(prev_frame_gray_, small_gray, prev_pts, curr_pts, status, err);

    std::vector<cv::Point2f> valid_prev, valid_curr;
    for (size_t i = 0; i < status.size(); ++i) {
      if (status[i]) {
        valid_prev.push_back(prev_pts[i]);
        valid_curr.push_back(curr_pts[i]);
      }
    }

    if (valid_prev.size() >= 6) {
      cv::Mat small_transform = cv::estimateAffinePartial2D(valid_prev, valid_curr);

      // FIX 2: Validate transformation matrix to prevent NaN / Inf coordinate corruption
      if (!small_transform.empty() && cv::checkRange(small_transform)) {
        // Rescale translation offsets to full image resolution
        small_transform.at<double>(0, 2) /= scale;
        small_transform.at<double>(1, 2) /= scale;
        transform = small_transform;
      }
    }
  }

  prev_frame_gray_ = small_gray.clone();
  return transform;
}

float TrackerEngine::computeIoU(const cv::Rect2f& boxA, const cv::Rect2f& boxB) {
  float x1 = std::max(boxA.x, boxB.x);
  float y1 = std::max(boxA.y, boxB.y);
  float x2 = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
  float y2 = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

  float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float boxAArea = boxA.width * boxA.height;
  float boxBArea = boxB.width * boxB.height;

  return interArea / (boxAArea + boxBArea - interArea + 1e-6f);
}

struct MatchPair {
  size_t track_idx;
  size_t det_idx;
  float iou;
};

std::vector<Track> TrackerEngine::update(
  const std::vector<opl_interfaces::msg::TrackedHuman>& detections, const cv::Mat& frame)
{
  // Step 1: Global Motion Compensation
  cv::Mat motion = estimateCameraMotion(frame);

  for (auto& track : active_tracks_) {
    track.time_since_update++;

    if (!motion.empty() && motion.cols == 3 && motion.rows == 2 && cv::checkRange(motion)) {
      // Full affine transformation on bounding box center coordinate
      cv::Point2f center(track.bbox.x + track.bbox.width * 0.5f, track.bbox.y + track.bbox.height * 0.5f);
      double a = motion.at<double>(0, 0);
      double b = motion.at<double>(0, 1);
      double tx = motion.at<double>(0, 2);
      double c = motion.at<double>(1, 0);
      double d = motion.at<double>(1, 1);
      double ty = motion.at<double>(1, 2);

      float new_cx = static_cast<float>(a * center.x + b * center.y + tx);
      float new_cy = static_cast<float>(c * center.x + d * center.y + ty);

      track.bbox.x = new_cx - track.bbox.width * 0.5f;
      track.bbox.y = new_cy - track.bbox.height * 0.5f;
    }
  }

  // Step 2: Global Ranked IoU Association
  std::vector<MatchPair> pairs;
  for (size_t i = 0; i < active_tracks_.size(); ++i) {
    for (size_t j = 0; j < detections.size(); ++j) {
      cv::Rect2f det_box(
        detections[j].bbox.x_offset, detections[j].bbox.y_offset,
        detections[j].bbox.width, detections[j].bbox.height);

      float iou = computeIoU(active_tracks_[i].bbox, det_box);
      if (iou >= iou_threshold_) {
        pairs.push_back({i, j, iou});
      }
    }
  }

  // Sort candidate pairs by IoU in descending order
  std::sort(pairs.begin(), pairs.end(), [](const MatchPair& a, const MatchPair& b) {
    return a.iou > b.iou;
  });

  std::vector<bool> det_matched(detections.size(), false);
  std::vector<bool> track_matched(active_tracks_.size(), false);

  for (const auto& pair : pairs) {
    if (track_matched[pair.track_idx] || det_matched[pair.det_idx]) continue;

    const auto& det = detections[pair.det_idx];
    active_tracks_[pair.track_idx].bbox = cv::Rect2f(
      det.bbox.x_offset, det.bbox.y_offset, det.bbox.width, det.bbox.height);
    active_tracks_[pair.track_idx].confidence = det.confidence;
    active_tracks_[pair.track_idx].hits++;
    active_tracks_[pair.track_idx].time_since_update = 0;

    track_matched[pair.track_idx] = true;
    det_matched[pair.det_idx] = true;
  }

  // Step 3: Instantiate New Tracks for Unmatched Detections
  for (size_t j = 0; j < detections.size(); ++j) {
    if (!det_matched[j]) {
      Track new_track;
      new_track.id = next_id_++;
      new_track.bbox = cv::Rect2f(
        detections[j].bbox.x_offset, detections[j].bbox.y_offset,
        detections[j].bbox.width, detections[j].bbox.height);
      new_track.confidence = detections[j].confidence;
      new_track.hits = 1;
      new_track.time_since_update = 0;
      active_tracks_.push_back(new_track);
    }
  }

  // Step 4: Track Lifecycle Filtering
  std::vector<Track> confirmed_tracks;
  auto it = active_tracks_.begin();
  while (it != active_tracks_.end()) {
    if (it->time_since_update > max_age_) {
      it = active_tracks_.erase(it);
    } else {
      if (it->hits >= min_hits_) {
        confirmed_tracks.push_back(*it);
      }
      ++it;
    }
  }

  return confirmed_tracks;
}

}  // namespace opl_human_vision
