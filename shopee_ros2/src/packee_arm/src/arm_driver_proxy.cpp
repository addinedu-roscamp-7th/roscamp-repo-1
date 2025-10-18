#include "packee_arm/arm_driver_proxy.hpp"

#include <cmath>
#include <limits>

#include "rclcpp/rclcpp.hpp"

namespace packee_arm {

ArmDriverProxy::ArmDriverProxy(
  const rclcpp::Logger & logger,
  double max_translation_speed,
  double max_yaw_speed_deg,
  double command_timeout_sec)
: logger_(logger),
  max_translation_speed_(max_translation_speed),
  max_yaw_speed_deg_(max_yaw_speed_deg),
  command_timeout_sec_(command_timeout_sec),
  last_command_time_(std::chrono::steady_clock::now()) {}


bool ArmDriverProxy::SendVelocityCommand(
  const std::string & arm_side,
  double vx,
  double vy,
  double vz,
  double yaw_rate_deg) {
  std::lock_guard<std::mutex> lock(mutex_);
  const double translation_norm = std::sqrt((vx * vx) + (vy * vy) + (vz * vz));
  if (translation_norm > max_translation_speed_ + std::numeric_limits<double>::epsilon()) {
    RCLCPP_WARN(
      logger_,
      "팔 %s 속도 명령이 최대 속도를 초과했습니다. 요청=%.3f, 제한=%.3f",
      arm_side.c_str(),
      translation_norm,
      max_translation_speed_);
    return false;
  }
  if (std::fabs(yaw_rate_deg) > max_yaw_speed_deg_ + std::numeric_limits<double>::epsilon()) {
    RCLCPP_WARN(
      logger_,
      "팔 %s yaw 속도가 제한을 초과했습니다. 요청=%.3f, 제한=%.3f",
      arm_side.c_str(),
      yaw_rate_deg,
      max_yaw_speed_deg_);
    return false;
  }
  last_command_time_ = std::chrono::steady_clock::now();
  return true;
}


bool ArmDriverProxy::HasTimedOut() const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto elapsed = std::chrono::steady_clock::now() - last_command_time_;
  return elapsed > std::chrono::duration<double>(command_timeout_sec_);
}


void ArmDriverProxy::UpdateConstraints(
  double max_translation_speed,
  double max_yaw_speed_deg,
  double command_timeout_sec) {
  std::lock_guard<std::mutex> lock(mutex_);
  max_translation_speed_ = max_translation_speed;
  max_yaw_speed_deg_ = max_yaw_speed_deg;
  command_timeout_sec_ = command_timeout_sec;
}

}  // namespace packee_arm

