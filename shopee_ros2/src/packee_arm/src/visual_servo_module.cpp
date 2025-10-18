#include "packee_arm/visual_servo_module.hpp"

#include <algorithm>
#include <cmath>

namespace packee_arm {

VisualServoModule::VisualServoModule(
  double servo_gain_xy,
  double servo_gain_z,
  double servo_gain_yaw,
  double confidence_threshold,
  double max_translation_speed,
  double max_yaw_speed_deg)
: servo_gain_xy_(servo_gain_xy),
  servo_gain_z_(servo_gain_z),
  servo_gain_yaw_(servo_gain_yaw),
  confidence_threshold_(confidence_threshold),
  max_translation_speed_(max_translation_speed),
  max_yaw_speed_deg_(max_yaw_speed_deg) {}


ServoCommand VisualServoModule::ComputeCommand(
  const PoseEstimate & current,
  const PoseEstimate & target) {
  std::lock_guard<std::mutex> lock(mutex_);
  const double error_x = target.x - current.x;
  const double error_y = target.y - current.y;
  const double error_z = target.z - current.z;
  const double error_yaw = target.yaw_deg - current.yaw_deg;

  const double raw_vx = error_x * servo_gain_xy_;
  const double raw_vy = error_y * servo_gain_xy_;
  const double raw_vz = error_z * servo_gain_z_;
  const double raw_yaw = error_yaw * servo_gain_yaw_;

  ServoCommand command{};
  command.vx = std::clamp(raw_vx, -max_translation_speed_, max_translation_speed_);
  command.vy = std::clamp(raw_vy, -max_translation_speed_, max_translation_speed_);
  command.vz = std::clamp(raw_vz, -max_translation_speed_, max_translation_speed_);
  command.yaw_rate_deg = std::clamp(raw_yaw, -max_yaw_speed_deg_, max_yaw_speed_deg_);
  command.confidence = target.confidence;
  command.error_norm = std::sqrt(
    (error_x * error_x) + (error_y * error_y) + (error_z * error_z) + (error_yaw * error_yaw));
  command.goal_reached =
    std::fabs(error_x) <= kTranslationTolerance &&
    std::fabs(error_y) <= kTranslationTolerance &&
    std::fabs(error_z) <= kTranslationTolerance &&
    std::fabs(error_yaw) <= kYawToleranceDeg;
  return command;
}


void VisualServoModule::UpdateGains(
  double servo_gain_xy,
  double servo_gain_z,
  double servo_gain_yaw) {
  std::lock_guard<std::mutex> lock(mutex_);
  servo_gain_xy_ = servo_gain_xy;
  servo_gain_z_ = servo_gain_z;
  servo_gain_yaw_ = servo_gain_yaw;
}


void VisualServoModule::UpdateConstraints(
  double confidence_threshold,
  double max_translation_speed,
  double max_yaw_speed_deg) {
  std::lock_guard<std::mutex> lock(mutex_);
  confidence_threshold_ = confidence_threshold;
  max_translation_speed_ = max_translation_speed;
  max_yaw_speed_deg_ = max_yaw_speed_deg;
}


double VisualServoModule::GetConfidenceThreshold() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return confidence_threshold_;
}

}  // namespace packee_arm

