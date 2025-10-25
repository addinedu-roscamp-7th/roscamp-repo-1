#ifndef PACKEE_ARM_ARM_DRIVER_PROXY_HPP_
#define PACKEE_ARM_ARM_DRIVER_PROXY_HPP_

#include <chrono>
#include <mutex>
#include <string>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/logger.hpp"

namespace packee_arm {

// ArmDriverProxy 클래스는 하드웨어 속도 명령 인터페이스를 추상화한다.
class ArmDriverProxy {
public:
  ArmDriverProxy(
    rclcpp::Node * node,
    double max_translation_speed,
    double max_yaw_speed_deg,
    double command_timeout_sec,
    const std::string & left_velocity_topic,
    const std::string & right_velocity_topic,
    const std::string & velocity_frame_id);

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
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr GetPublisher(
    const std::string & arm_side);

  void PublishTwist(
    const rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr & publisher,
    double vx,
    double vy,
    double vz,
    double yaw_rate_deg);

  rclcpp::Node * node_;
  rclcpp::Logger logger_;
  mutable std::mutex mutex_;
  double max_translation_speed_;
  double max_yaw_speed_deg_;
  double command_timeout_sec_;
  std::chrono::steady_clock::time_point last_command_time_;
  std::string velocity_frame_id_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr left_velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr right_velocity_pub_;
};

}  // namespace packee_arm

#endif  // PACKEE_ARM_ARM_DRIVER_PROXY_HPP_
