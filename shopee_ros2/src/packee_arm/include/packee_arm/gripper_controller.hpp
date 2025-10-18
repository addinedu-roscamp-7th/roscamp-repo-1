#ifndef PACKEE_ARM_GRIPPER_CONTROLLER_HPP_
#define PACKEE_ARM_GRIPPER_CONTROLLER_HPP_

#include <mutex>
#include <string>

#include "rclcpp/logger.hpp"

namespace packee_arm {

// GripperController 클래스는 그리퍼 힘 제어를 담당한다.
class GripperController {
public:
  GripperController(const rclcpp::Logger & logger, double force_limit);

  bool Close(const std::string & arm_side, double requested_force);

  bool Open(const std::string & arm_side);

  void UpdateForceLimit(double force_limit);

  double GetForceLimit() const;

private:
  rclcpp::Logger logger_;
  mutable std::mutex mutex_;
  double force_limit_;
};

}  // namespace packee_arm

#endif  // PACKEE_ARM_GRIPPER_CONTROLLER_HPP_
