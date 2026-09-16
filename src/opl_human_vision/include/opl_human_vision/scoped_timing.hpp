#pragma once

#include <chrono>
#include <rclcpp/rclcpp.hpp>

namespace opl_human_vision {

// Wall time includes preprocessing, GPU transfers/synchronization and postprocessing.
// Nested timings overlap; do not add module times to their enclosing callback time.
class ScopedTiming {
public:
  explicit ScopedTiming(const char* module)
  : module_(module), start_(std::chrono::steady_clock::now()) {}

  ~ScopedTiming() {
    const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start_).count();
    RCLCPP_INFO(rclcpp::get_logger("vision_timing"), "[timing] %s wall_ms=%.3f", module_, ms);
  }

private:
  const char* module_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace opl_human_vision
