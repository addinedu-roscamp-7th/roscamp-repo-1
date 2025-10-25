#include "packee_arm/arm_driver_proxy.hpp"

#include <cmath>
#include <limits>

#include "rclcpp/rclcpp.hpp"

namespace packee_arm {

ArmDriverProxy::ArmDriverProxy(
  rclcpp::Node * node,
  double max_translation_speed,
  double max_yaw_speed_deg,
  double command_timeout_sec,
  const std::string & left_velocity_topic,
  const std::string & right_velocity_topic,
  const std::string & velocity_frame_id)
: node_(node),
  logger_(node_->get_logger().get_child("ArmDriverProxy")),
  max_translation_speed_(max_translation_speed),
  max_yaw_speed_deg_(max_yaw_speed_deg),
  command_timeout_sec_(command_timeout_sec),
  last_command_time_(std::chrono::steady_clock::now()),
  velocity_frame_id_(velocity_frame_id) {
  // JetCobot 물리 팔에 속도 명령을 전달할 퍼블리셔를 준비한다.
  left_velocity_pub_ = node_->create_publisher<geometry_msgs::msg::TwistStamped>(
    left_velocity_topic,
    rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
  right_velocity_pub_ = node_->create_publisher<geometry_msgs::msg::TwistStamped>(
    right_velocity_topic,
    rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
}


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
  auto publisher = GetPublisher(arm_side);
  if (!publisher) {
    RCLCPP_ERROR(logger_, "팔 %s 토픽 퍼블리셔가 설정되지 않았습니다.", arm_side.c_str());
    return false;
  }
  PublishTwist(publisher, vx, vy, vz, yaw_rate_deg);
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


rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr ArmDriverProxy::GetPublisher(
  const std::string & arm_side) {
  if ("left" == arm_side) {
    return left_velocity_pub_;
  }
  if ("right" == arm_side) {
    return right_velocity_pub_;
  }
  return nullptr;
}


void ArmDriverProxy::PublishTwist(
  const rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr & publisher,
  double vx,
  double vy,
  double vz,
  double yaw_rate_deg) {
  // JetCobot 브리지가 직접 사용할 수 있도록 TwistStamped 형태로 명령을 구성한다.
  auto twist = geometry_msgs::msg::TwistStamped();
  twist.header.stamp = node_->now();
  twist.header.frame_id = velocity_frame_id_;
  twist.twist.linear.x = vx;
  twist.twist.linear.y = vy;
  twist.twist.linear.z = vz;
  twist.twist.angular.x = 0.0;
  twist.twist.angular.y = 0.0;
  twist.twist.angular.z = yaw_rate_deg * (M_PI / 180.0);
  publisher->publish(twist);
}

}  // namespace packee_arm
