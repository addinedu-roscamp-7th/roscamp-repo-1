#ifndef PACKEE_ARM_ARM_DRIVER_PROXY_HPP_
#define PACKEE_ARM_ARM_DRIVER_PROXY_HPP_

#include <chrono>
#include <mutex>
#include <string>

#include "rclcpp/logger.hpp"

namespace packee_arm {

// ArmDriverProxy 클래스는 하드웨어 속도 명령 인터페이스를 추상화한다.
class ArmDriverProxy {
public:
  ArmDriverProxy(
    const rclcpp::Logger & logger,
    double max_translation_speed,
    double max_yaw_speed_deg,
    double command_timeout_sec);

  bool SendVelocityCommand(
    const std::string & arm_side,
    double vx,
    double vy,
    double vz,
    double yaw_rate_deg);

  bool HasTimedOut() const;

  void UpdateConstraints(
    double max_translation_speed,
    double max_yaw_speed_deg,
    double command_timeout_sec);

private:
  rclcpp::Logger logger_;
  mutable std::mutex mutex_;
  double max_translation_speed_;
  double max_yaw_speed_deg_;
  double command_timeout_sec_;
  std::chrono::steady_clock::time_point last_command_time_;
};

}  // namespace packee_arm

#endif  // PACKEE_ARM_ARM_DRIVER_PROXY_HPP_
