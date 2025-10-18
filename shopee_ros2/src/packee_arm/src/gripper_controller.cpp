#include "packee_arm/gripper_controller.hpp"

#include <algorithm>

#include "rclcpp/rclcpp.hpp"

namespace packee_arm {

GripperController::GripperController(const rclcpp::Logger & logger, double force_limit)
: logger_(logger), force_limit_(force_limit) {}


bool GripperController::Close(const std::string & arm_side, double requested_force) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (requested_force > force_limit_) {
    RCLCPP_ERROR(
      logger_,
      "팔 %s 그리퍼 힘이 한계를 초과했습니다. 요청=%.2f, 제한=%.2f",
      arm_side.c_str(),
      requested_force,
      force_limit_);
    return false;
  }
  return true;
}


bool GripperController::Open(const std::string & arm_side) {
  std::lock_guard<std::mutex> lock(mutex_);
  (void)arm_side;
  return true;
}


void GripperController::UpdateForceLimit(double force_limit) {
  std::lock_guard<std::mutex> lock(mutex_);
  force_limit_ = force_limit;
}


double GripperController::GetForceLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return force_limit_;
}

}  // namespace packee_arm

