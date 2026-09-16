#include "opl_human_vision/face_recognizer_node.hpp"
#include "opl_human_vision/scoped_timing.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <opl_interfaces/srv/enroll_person.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cv_bridge/cv_bridge.hpp>

namespace opl_human_vision {

FaceRecognizerNode::FaceRecognizerNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("face_recognizer_node", options)
{
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

  // How long a raw-track -> persistent-ID lock is kept alive
  // after the raw track temporarily disappears from published output.
  declare_parameter<double>("identity_lock_grace_sec", 1.0);
  declare_parameter<double>("identity_validation_interval_sec", 0.5);
  declare_parameter<double>("session_reid_timeout_sec", 5.0);
  declare_parameter<double>("reid_spatial_gate_max_age_sec",2.0);
  declare_parameter<double>("reid_spatial_gate_base_m",0.75);
  declare_parameter<double>("reid_spatial_gate_max_speed_mps",2.5);
  declare_parameter<double>("reid_depth_gate_base_m",0.75);
}


rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(
    get_logger(),
    "Configuring FaceRecognizerNode..."
  );

  image_topic_ =
    get_parameter("image_topic").as_string();

  tracked_topic_ =
    get_parameter("tracked_topic").as_string();

  recognized_topic_ =
    get_parameter("recognized_topic").as_string();

  identity_lock_grace_sec_ =
    get_parameter("identity_lock_grace_sec").as_double();

  identity_validation_interval_sec_ =
    get_parameter(
      "identity_validation_interval_sec"
    ).as_double();

  if (
    identity_validation_interval_sec_ <
    0.0)
  {
    identity_validation_interval_sec_ =
      0.0;
  }

  const double session_reid_timeout =
    get_parameter("session_reid_timeout_sec").as_double();

  const std::string scrfd_path =
    get_parameter("scrfd_model_path").as_string();

  const std::string adaface_path =
    get_parameter("adaface_model_path").as_string();

  const std::string osnet_path =
    get_parameter("osnet_model_path").as_string();


  const float match_thresh =
    static_cast<float>(
      get_parameter("match_threshold").as_double()
    );

  const float body_thresh =
    static_cast<float>(
      get_parameter("body_match_threshold").as_double()
    );

  const int min_confirm_frames =
    get_parameter("min_confirm_frames").as_int();

  const float min_confirm_duration =
    static_cast<float>(
      get_parameter("min_confirm_duration_sec").as_double()
    );
  const double spatial_gate_max_age =
  get_parameter(
    "reid_spatial_gate_max_age_sec"
  ).as_double();

  const float spatial_gate_base =
    static_cast<float>(
      get_parameter(
        "reid_spatial_gate_base_m"
      ).as_double()
    );

  const float spatial_gate_max_speed =
    static_cast<float>(
      get_parameter(
        "reid_spatial_gate_max_speed_mps"
      ).as_double()
    );

  const float depth_gate_base =
    static_cast<float>(
      get_parameter(
        "reid_depth_gate_base_m"
      ).as_double()
    );

  face_engine_ = std::make_shared<FaceEngine>(
    scrfd_path,
    adaface_path,
    osnet_path,
    match_thresh,
    body_thresh,
    min_confirm_frames,
    min_confirm_duration,
    session_reid_timeout,
    spatial_gate_max_age,
    spatial_gate_base,
    spatial_gate_max_speed,
    depth_gate_base
  );

  if (!face_engine_->loadModels()) {
    RCLCPP_ERROR(
      get_logger(),
      "Failed to initialize TensorRT models in FaceEngine!"
    );

    return CallbackReturn::FAILURE;
  }


  recognized_pub_ =
    create_publisher<opl_interfaces::msg::TrackedHumanArray>(
      recognized_topic_,
      rclcpp::SensorDataQoS()
    );


  enroll_srv_ =
    create_service<opl_interfaces::srv::EnrollPerson>(
      "~/enroll_person",
      std::bind(
        &FaceRecognizerNode::handleEnrollService,
        this,
        std::placeholders::_1,
        std::placeholders::_2
      )
    );


  RCLCPP_INFO(
    get_logger(),
    "Identity lock grace period: %.2f sec",
    identity_lock_grace_sec_
  );

  RCLCPP_INFO(
    get_logger(),
    "FaceRecognizerNode configured successfully."
  );

  return CallbackReturn::SUCCESS;
}


rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_activate(
  const rclcpp_lifecycle::State & state)
{
  LifecycleNode::on_activate(state);

  recognized_pub_->on_activate();


  image_cb_group_ =
    create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive
    );

  tracked_cb_group_ =
    create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive
    );


  rclcpp::SubscriptionOptions img_sub_options;

  img_sub_options.callback_group =
    image_cb_group_;

  img_sub_options.use_intra_process_comm =
    rclcpp::IntraProcessSetting::Enable;


  image_sub_ =
    create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(
        &FaceRecognizerNode::imageCallback,
        this,
        std::placeholders::_1
      ),
      img_sub_options
    );


  rclcpp::SubscriptionOptions tracked_sub_options;

  tracked_sub_options.callback_group =
    tracked_cb_group_;

  tracked_sub_options.use_intra_process_comm =
    rclcpp::IntraProcessSetting::Enable;


  tracked_sub_ =
    create_subscription<opl_interfaces::msg::TrackedHumanArray>(
      tracked_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(
        &FaceRecognizerNode::trackedHumansCallback,
        this,
        std::placeholders::_1
      ),
      tracked_sub_options
    );


  RCLCPP_INFO(
    get_logger(),
    "FaceRecognizerNode activated."
  );

  return CallbackReturn::SUCCESS;
}


rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_deactivate(
  const rclcpp_lifecycle::State & state)
{
  LifecycleNode::on_deactivate(state);

  image_sub_.reset();
  tracked_sub_.reset();

  recognized_pub_->on_deactivate();

  return CallbackReturn::SUCCESS;
}


rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  recognized_pub_.reset();
  enroll_srv_.reset();
  face_engine_.reset();


  {
    std::lock_guard<std::mutex> lock(state_mutex_);

    currently_visible_pids_.clear();
    ever_scanned_pids_.clear();
    pid_to_name_.clear();
    locked_tracks_.clear();
    last_active_raw_ids_.clear();
  }


  {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    frame_buffer_.clear();
  }


  return CallbackReturn::SUCCESS;
}


void FaceRecognizerNode::handleEnrollService(
  const std::shared_ptr<
    opl_interfaces::srv::EnrollPerson::Request> request,
  std::shared_ptr<
    opl_interfaces::srv::EnrollPerson::Response> response)
{
  uint64_t target_id =
    request->raw_track_id;


  // If raw_track_id == 0, use one of the currently active
  // unassigned/raw tracks as a fallback.
  if (target_id == 0) {

    std::lock_guard<std::mutex> lock(state_mutex_);

    if (!last_active_raw_ids_.empty()) {
      target_id =
        *last_active_raw_ids_.begin();
    }
  }


  if (target_id == 0) {

    response->success = false;

    response->message =
      "No active track found to authorize enrollment.";

    return;
  }


  {
    std::lock_guard<std::mutex> gpu_lock(
      engine_mutex_
    );

    face_engine_->authorizeEnrollment(
      target_id,
      request->name
    );
  }

  {
    std::lock_guard<std::mutex> state_lock(
      state_mutex_
    );

    locked_tracks_.erase(
      target_id
    );
  }

  response->success = true;

  response->message =
    "Enrollment authorized for Raw Track ID " +
    std::to_string(target_id);


  RCLCPP_INFO(
    get_logger(),
    "Enrollment authorized for raw track %lu",
    target_id
  );
}


void FaceRecognizerNode::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  try {

    auto cv_ptr =
      cv_bridge::toCvShare(
        msg,
        sensor_msgs::image_encodings::BGR8
      );


    std::lock_guard<std::mutex> lock(
      frame_mutex_
    );


    frame_buffer_.emplace_back(
      rclcpp::Time(msg->header.stamp),
      cv_ptr->image
    );


    // Bound memory use.
    while (frame_buffer_.size() > 30) {
      frame_buffer_.pop_front();
    }

  } catch (const cv_bridge::Exception & e) {

    RCLCPP_ERROR(
      get_logger(),
      "cv_bridge exception: %s",
      e.what()
    );
  }
}


void FaceRecognizerNode::trackedHumansCallback(
  const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg)
{
  if (!recognized_pub_->is_activated()) {
    return;
  }


  ScopedTiming timing(
    "Recognizer.callback_total"
  );


  // --------------------------------------------------------------------------
  // Step 1: Find camera frame closest to tracker timestamp.
  // --------------------------------------------------------------------------

  cv::Mat frame_ref;

  const rclcpp::Time target_time(
    msg->header.stamp
  );

  double min_diff =
    1e9;


  {
    std::lock_guard<std::mutex> lock(
      frame_mutex_
    );


    if (frame_buffer_.empty()) {

      RCLCPP_INFO(
        rclcpp::get_logger("vision_timing"),
        "[timing] Recognizer skipped: image buffer empty"
      );


      recognized_pub_->publish(
        *msg
      );

      return;
    }


    // Remove images that are much older than the tracker frame.
    while (
      !frame_buffer_.empty() &&
      (
        target_time -
        frame_buffer_.front().first
      ).seconds() > 0.20)
    {
      frame_buffer_.pop_front();
    }


    // Find closest RGB timestamp.
    for (const auto& item : frame_buffer_) {

      const double diff =
        std::abs(
          (
            item.first -
            target_time
          ).seconds()
        );


      if (diff < min_diff) {

        min_diff =
          diff;

        frame_ref =
          item.second;
      }
    }
  }


  // Do not run recognition on a badly mismatched RGB image.
  if (
    frame_ref.empty() ||
    min_diff > 0.10)
  {
    RCLCPP_INFO(
      rclcpp::get_logger("vision_timing"),
      "[timing] Recognizer skipped: "
      "no image within 100ms, nearest_diff_ms=%.3f",
      min_diff * 1000.0
    );


    recognized_pub_->publish(
      *msg
    );

    return;
  }


  // --------------------------------------------------------------------------
  // Step 2: Determine RAW tracker IDs visible this frame.
  // --------------------------------------------------------------------------

  opl_interfaces::msg::TrackedHumanArray
    recognized_array_msg;

  recognized_array_msg.header =
    msg->header;


  // IDs actually assigned while processing this frame.
  std::set<uint64_t>
    assigned_ids_in_frame;


  // Persistent/session IDs visible in final output.
  std::set<uint64_t>
    current_frame_pids;


  // Raw TrackerEngine IDs visible in this input message.
  std::set<uint64_t>
    active_raw_ids;


  // --------------------------------------------------------------------------
  // CRITICAL:
  //
  // Identity -> raw-track ownership.
  //
  // Example:
  //
  //     persistent/session ID 5 -> raw track 12
  //
  // Once an identity is owned by one active raw track, another active raw
  // track is not allowed to claim it.
  //
  // This removes dependence on the order in which msg->humans is processed.
  // --------------------------------------------------------------------------

  std::unordered_map<uint64_t, uint64_t>
    identity_owner;


  for (const auto& human : msg->humans) {

    uint64_t raw_id =
      0;


    if (human.raw_track_id > 0) {

      raw_id =
        static_cast<uint64_t>(
          human.raw_track_id
        );

    } else if (human.track_id > 0) {

      // Compatibility with older tracker output.
      raw_id =
        static_cast<uint64_t>(
          human.track_id
        );
    }


    if (raw_id > 0) {
      active_raw_ids.insert(
        raw_id
      );
    }
  }


  {
    std::lock_guard<std::mutex> state_lock(
      state_mutex_
    );


    last_active_raw_ids_ =
      active_raw_ids;
  }


  // --------------------------------------------------------------------------
  // Step 3: Clean FaceEngine raw-track mappings.
  // --------------------------------------------------------------------------

  {
    std::lock_guard<std::mutex> engine_lock(
      engine_mutex_
    );


    face_engine_->cleanupStaleRawTracks(
      active_raw_ids
    );
  }


  // --------------------------------------------------------------------------
  // Step 4: Maintain node-level identity locks.
  // --------------------------------------------------------------------------

  const auto callback_now =
    std::chrono::steady_clock::now();


  {
    std::lock_guard<std::mutex> state_lock(
      state_mutex_
    );


    // Refresh locks belonging to currently visible raw tracks.
    for (const uint64_t raw_id : active_raw_ids) {

      auto lock_it =
        locked_tracks_.find(
          raw_id
        );


      if (
        lock_it !=
        locked_tracks_.end())
      {
        lock_it->second.last_seen =
          callback_now;
      }
    }


    // Expire locks only after grace period.
    for (
      auto it =
        locked_tracks_.begin();

      it !=
        locked_tracks_.end();)
    {
      const bool currently_visible =
        active_raw_ids.find(
          it->first
        ) !=
        active_raw_ids.end();


      if (currently_visible) {

        ++it;

        continue;
      }


      const double missing_sec =
        std::chrono::duration<double>(
          callback_now -
          it->second.last_seen
        ).count();


      if (
        missing_sec >
        identity_lock_grace_sec_)
      {
        RCLCPP_INFO(
          get_logger(),
          "Identity lock expired after %.2fs: "
          "Raw Track %lu -> "
          "Persistent ID %lu ('%s')",
          missing_sec,
          it->first,
          it->second.persistent_id,
          it->second.name.c_str()
        );


        it =
          locked_tracks_.erase(
            it
          );

      } else {

        ++it;
      }
    }


    // ------------------------------------------------------------------------
    // Reserve identities belonging to currently ACTIVE locked tracks.
    //
    // active_raw_ids is std::set, so ownership selection is deterministic.
    //
    // This also repairs a legacy situation where two existing locks already
    // point to the same identity.
    // ------------------------------------------------------------------------

    std::vector<uint64_t>
      conflicting_raw_locks;


    for (const uint64_t raw_id : active_raw_ids) {

      auto lock_it =
        locked_tracks_.find(
          raw_id
        );


      if (
        lock_it ==
        locked_tracks_.end())
      {
        continue;
      }


      const uint64_t identity_id =
        lock_it->second.persistent_id;


      if (identity_id == 0) {
        continue;
      }


      auto owner_it =
        identity_owner.find(
          identity_id
        );


      if (
        owner_it ==
        identity_owner.end())
      {
        // First active raw track reserves this identity.
        identity_owner[
          identity_id
        ] = raw_id;

      } else if (
        owner_it->second !=
        raw_id)
      {
        // Existing duplicate lock from an older frame/state.
        //
        // Keep the first deterministic owner and invalidate the duplicate
        // node-level lock. That raw track will go through normal ReID below.
        RCLCPP_WARN(
          get_logger(),
          "Identity ownership conflict: "
          "Persistent ID %lu already owned by Raw Track %lu; "
          "dropping duplicate lock from Raw Track %lu",
          identity_id,
          owner_it->second,
          raw_id
        );


        conflicting_raw_locks.push_back(
          raw_id
        );
      }
    }


    for (
      const uint64_t raw_id :
      conflicting_raw_locks)
    {
      locked_tracks_.erase(
        raw_id
      );
    }
  }


  // --------------------------------------------------------------------------
  // Step 5: Process each visible raw track.
  // --------------------------------------------------------------------------

  for (auto human : msg->humans) {

    // ------------------------------------------------------------------------
    // Resolve raw tracker ID.
    // ------------------------------------------------------------------------

    uint64_t raw_track_id =
      0;


    if (human.raw_track_id > 0) {

      raw_track_id =
        static_cast<uint64_t>(
          human.raw_track_id
        );

    } else if (human.track_id > 0) {

      raw_track_id =
        static_cast<uint64_t>(
          human.track_id
        );
    }


    if (raw_track_id == 0) {

      recognized_array_msg.humans.push_back(
        human
      );

      continue;
    }


    // Always preserve real TrackerEngine ID.
    human.raw_track_id =
      static_cast<int32_t>(
        raw_track_id
      );


    // Fallback until persistent/session recognition succeeds.
    human.track_id =
      static_cast<int32_t>(
        raw_track_id
      );

    // ------------------------------------------------------------------------
    // Check node-level lock.
    // ------------------------------------------------------------------------

    bool is_locked =
      false;

    bool validation_due =
      false;

    LockedTrackInfo
      locked_info;


    {
      std::lock_guard<std::mutex> state_lock(
        state_mutex_
      );


      auto lock_it =
        locked_tracks_.find(
          raw_track_id
        );


      if (
        lock_it !=
        locked_tracks_.end())
      {
        const uint64_t identity_id =
          lock_it->second.persistent_id;


        auto owner_it =
          identity_owner.find(
            identity_id
          );


        if (
          owner_it !=
            identity_owner.end() &&
          owner_it->second ==
            raw_track_id)
        {
          is_locked =
            true;

          locked_info =
            lock_it->second;


          const double time_since_validation =
            std::chrono::duration<double>(
              callback_now -
              locked_info.last_validation
            ).count();


          validation_due =
            identity_validation_interval_sec_ <= 0.0 ||
            time_since_validation >=
              identity_validation_interval_sec_;
        }
      }
    }


    // ------------------------------------------------------------------------
    // FAST PATH
    //
    // Existing raw-track lock.
    // ------------------------------------------------------------------------

    if (is_locked && !validation_due) {

      human.raw_track_id =
        static_cast<int32_t>(
          raw_track_id
        );


      human.track_id =
        static_cast<int32_t>(
          locked_info.persistent_id
        );


      human.name =
        locked_info.name;


      // Keep spatial state fresh even while expensive ReID is bypassed.
      {
        std::lock_guard<std::mutex> engine_lock(
          engine_mutex_
        );


        face_engine_->updateIdentityPosition(
          locked_info.persistent_id,
          human.position_3d
        );
      }


      assigned_ids_in_frame.insert(
        locked_info.persistent_id
      );


      current_frame_pids.insert(
        locked_info.persistent_id
      );
    }


    // ------------------------------------------------------------------------
    // SLOW PATH
    //
    // No valid node-level identity lock.
    // ------------------------------------------------------------------------

    else {
      const bool force_deep_reid =
        is_locked &&
        validation_due;


      const uint64_t previous_locked_id =
        is_locked
        ? locked_info.persistent_id
        : 0;

      const int raw_x =
        static_cast<int>(
          human.bbox.x_offset
        );

      const int raw_y =
        static_cast<int>(
          human.bbox.y_offset
        );

      const int raw_w =
        static_cast<int>(
          human.bbox.width
        );

      const int raw_h =
        static_cast<int>(
          human.bbox.height
        );


      const int x1 =
        std::max(
          0,
          std::min(
            raw_x,
            frame_ref.cols - 1
          )
        );


      const int y1 =
        std::max(
          0,
          std::min(
            raw_y,
            frame_ref.rows - 1
          )
        );


      const int x2 =
        std::max(
          0,
          std::min(
            raw_x + raw_w,
            frame_ref.cols
          )
        );


      const int y2 =
        std::max(
          0,
          std::min(
            raw_y + raw_h,
            frame_ref.rows
          )
        );


      const int crop_w =
        x2 - x1;

      const int crop_h =
        y2 - y1;


      if (
        crop_w > 0 &&
        crop_h > 0)
      {
        const cv::Rect human_rect(
          x1,
          y1,
          crop_w,
          crop_h
        );


        const cv::Mat human_crop =
          frame_ref(
            human_rect
          ).clone();


        // --------------------------------------------------------------------
        // Construct occupied identity set specifically for THIS raw track.
        //
        // It contains:
        //
        //   1. identities already assigned earlier this frame
        //   2. identities reserved by OTHER active locked raw tracks
        //
        // The raw track's own identity, if any, is not blocked.
        // --------------------------------------------------------------------

        std::set<uint64_t>
          occupied_ids_for_raw =
            assigned_ids_in_frame;


        for (
          const auto& owner_pair :
          identity_owner)
        {
          const uint64_t identity_id =
            owner_pair.first;

          const uint64_t owner_raw_id =
            owner_pair.second;


          if (
            owner_raw_id !=
            raw_track_id)
          {
            occupied_ids_for_raw.insert(
              identity_id
            );
          }
        }


        FaceResult result;


        {
          ScopedTiming
            lock_and_process_timing(
              "Recognizer.engine_lock_and_process"
            );


          std::lock_guard<std::mutex> engine_lock(
            engine_mutex_
          );


          result =
            face_engine_->processFace(
              human_crop,
              raw_track_id,
              occupied_ids_for_raw,
              human.position_3d,
              force_deep_reid
            );
        }


        // --------------------------------------------------------------------
        // Convert crop-relative face landmarks to image coordinates.
        // --------------------------------------------------------------------

        human.landmarks.clear();


        for (const auto& lm : result.landmarks) {

          geometry_msgs::msg::Point pt;


          pt.x =
            static_cast<double>(
              x1
            ) +
            lm.x;


          pt.y =
            static_cast<double>(
              y1
            ) +
            lm.y;


          pt.z =
            0.0;


          human.landmarks.push_back(
            pt
          );
        }


        human.raw_track_id =
          static_cast<int32_t>(
            raw_track_id
          );


        // --------------------------------------------------------------------
        // Final ownership guard.
        //
        // FaceEngine normally respects occupied_ids_for_raw during gallery
        // search. This extra node-level guard is deliberate because cached
        // raw-track mappings inside FaceEngine may return before a gallery
        // search occurs.
        // --------------------------------------------------------------------

        bool identity_claim_allowed =
          false;


        if (result.persistent_id > 0) {

          auto owner_it =
            identity_owner.find(
              result.persistent_id
            );


          const bool owned_by_other =
            owner_it !=
              identity_owner.end() &&
            owner_it->second !=
              raw_track_id;


          const bool assigned_by_other =
            assigned_ids_in_frame.find(
              result.persistent_id
            ) !=
              assigned_ids_in_frame.end() &&
            (
              owner_it ==
                identity_owner.end() ||
              owner_it->second !=
                raw_track_id
            );


          if (
            !owned_by_other &&
            !assigned_by_other)
          {
            identity_claim_allowed =
              true;


            // If periodic validation changed this raw track from
            // one identity to another, release the old reservation.
            if (
              force_deep_reid &&
              previous_locked_id > 0 &&
              previous_locked_id !=
                result.persistent_id)
            {
              auto previous_owner_it =
                identity_owner.find(
                  previous_locked_id
                );


              if (
                previous_owner_it !=
                  identity_owner.end() &&
                previous_owner_it->second ==
                  raw_track_id)
              {
                identity_owner.erase(
                  previous_owner_it
                );
              }
            }


            identity_owner[
              result.persistent_id
            ] =
              raw_track_id;

          } else {

            const uint64_t existing_owner =
              owner_it !=
                identity_owner.end()
              ? owner_it->second
              : 0;


            RCLCPP_WARN(
              get_logger(),
              "Rejected duplicate identity assignment: "
              "Raw Track %lu attempted Persistent ID %lu "
              "already owned by Raw Track %lu",
              raw_track_id,
              result.persistent_id,
              existing_owner
            );
          }
        }


        if (
          force_deep_reid &&
          (
            result.persistent_id == 0 ||
            !identity_claim_allowed
          ))

        {
          {
            std::lock_guard<std::mutex> state_lock(
              state_mutex_
            );


            locked_tracks_.erase(
              raw_track_id
            );
          }


          auto old_owner_it =
            identity_owner.find(
              previous_locked_id
            );


          if (
            old_owner_it !=
              identity_owner.end() &&
            old_owner_it->second ==
              raw_track_id)
          {
            identity_owner.erase(
              old_owner_it
            );
          }


          RCLCPP_WARN(
            get_logger(),
            "Identity validation failed: "
            "Raw Track %lu released Persistent ID %lu",
            raw_track_id,
            previous_locked_id
          );
        }

        // --------------------------------------------------------------------
        // Accepted persistent/session identity.
        // --------------------------------------------------------------------

        if (
          result.persistent_id > 0 &&
          identity_claim_allowed)
        {
          human.track_id =
            static_cast<int32_t>(
              result.persistent_id
            );


          human.name =
            result.name;


          human.match_score =
            result.match_score;


          assigned_ids_in_frame.insert(
            result.persistent_id
          );


          current_frame_pids.insert(
            result.persistent_id
          );


          {
            std::lock_guard<std::mutex> state_lock(
              state_mutex_
            );


            pid_to_name_[
              result.persistent_id
            ] =
              result.name;


            if (result.just_scanned) {

              ever_scanned_pids_.insert(
                result.persistent_id
              );
            }
          }

        } else {

          // ---------------------------------------------------------------
          // No usable identity.
          //
          // This also handles a ReID result rejected by the one-owner rule.
          // Never publish somebody else's identity just because FaceEngine
          // returned it from a cached mapping.
          // ---------------------------------------------------------------

          human.track_id =
            static_cast<int32_t>(
              raw_track_id
            );


          if (
            result.persistent_id > 0 &&
            !identity_claim_allowed)
          {
            human.name =
              "unknown";

            human.match_score =
              0.0f;

          } else {

            human.name =
              result.name;

            human.match_score =
              result.match_score;
          }
        }


        // --------------------------------------------------------------------
        // Determine whether accepted identity is stable enough to lock.
        // --------------------------------------------------------------------

        std::string name_lower =
          result.name;


        std::transform(
          name_lower.begin(),
          name_lower.end(),
          name_lower.begin(),

          [](unsigned char c)
          {
            return static_cast<char>(
              std::tolower(c)
            );
          }
        );


        const bool is_scanning =
          name_lower.find(
            "scanning"
          ) !=
          std::string::npos;


        const bool is_unregistered =
          name_lower.find(
            "unregistered"
          ) !=
          std::string::npos;


        // --------------------------------------------------------------------
        // Lock only if:
        //
        //   * FaceEngine returned an identity
        //   * node-level ownership accepted it
        //   * enrollment isn't still scanning
        //   * identity isn't an unregistered placeholder
        // --------------------------------------------------------------------

        if (
          identity_claim_allowed &&
          !result.name.empty() &&
          !is_scanning &&
          !is_unregistered &&
          result.persistent_id > 0)
        {
          std::lock_guard<std::mutex> state_lock(
            state_mutex_
          );


          // Recheck ownership before committing the lock.
          auto owner_it =
            identity_owner.find(
              result.persistent_id
            );


          if (
            owner_it !=
              identity_owner.end() &&
            owner_it->second ==
              raw_track_id)
          {
            LockedTrackInfo new_lock;

            new_lock.name =
              result.name;

            new_lock.persistent_id =
              result.persistent_id;

            new_lock.last_seen =
              callback_now;

            new_lock.last_validation =
              callback_now;

            locked_tracks_[
              raw_track_id
            ] =
              new_lock;

            RCLCPP_INFO(
              get_logger(),
              "Identity Locked: "
              "Raw Track %lu -> "
              "Persistent ID %lu ('%s')",
              raw_track_id,
              result.persistent_id,
              result.name.c_str()
            );
          }
        }
      }
    }


    recognized_array_msg.humans.push_back(
      human
    );
  }


  // --------------------------------------------------------------------------
  // Step 6: Save identities currently visible.
  // --------------------------------------------------------------------------

  {
    std::lock_guard<std::mutex> state_lock(
      state_mutex_
    );


    currently_visible_pids_ =
      current_frame_pids;
  }


  // --------------------------------------------------------------------------
  // Step 7: Publish.
  // --------------------------------------------------------------------------

  recognized_pub_->publish(
    recognized_array_msg
  );
}



}  // namespace opl_human_vision


RCLCPP_COMPONENTS_REGISTER_NODE(
  opl_human_vision::FaceRecognizerNode
)