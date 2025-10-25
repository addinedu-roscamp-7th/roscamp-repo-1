#include "packee_arm/gripper_controller.hpp"

#include <algorithm>

#include "rclcpp/rclcpp.hpp"

namespace packee_arm {

GripperController::GripperController(
  rclcpp::Node * node,
  const rclcpp::Logger & logger,
  double force_limit,
  const std::string & left_gripper_topic,
  const std::string & right_gripper_topic)
: node_(node),
  logger_(logger),
  force_limit_(force_limit) {
  // JetCobot 그리퍼 명령을 전송할 퍼블리셔를 초기화한다.
  left_gripper_pub_ = node_->create_publisher<std_msgs::msg::Float32>(
    left_gripper_topic,
    rclcpp::QoS(rclcpp::KeepLast(5)).reliable());
  right_gripper_pub_ = node_->create_publisher<std_msgs::msg::Float32>(
    right_gripper_topic,
    rclcpp::QoS(rclcpp::KeepLast(5)).reliable());
}


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
  PublishGripperCommand(arm_side, requested_force);
  return true;
}


bool GripperController::Open(const std::string & arm_side) {
  std::lock_guard<std::mutex> lock(mutex_);
  // 0 N 명령을 발행하면 JetCobot 브리지가 자동으로 개방 명령으로 해석한다.
  PublishGripperCommand(arm_side, 0.0);
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


void GripperController::PublishGripperCommand(
  const std::string & arm_side,
  double force_newton) {
  auto publisher = GetPublisher(arm_side);
  if (!publisher) {
    RCLCPP_ERROR(logger_, "팔 %s 그리퍼 퍼블리셔가 설정되지 않았습니다.", arm_side.c_str());
    return;
  }
  auto message = std_msgs::msg::Float32();
  message.data = static_cast<float>(force_newton);
  publisher->publish(message);
}


rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr GripperController::GetPublisher(
  const std::string & arm_side) {
  if ("left" == arm_side) {
    return left_gripper_pub_;
  }
  if ("right" == arm_side) {
    return right_gripper_pub_;
  }
  return nullptr;
}

}  // namespace packee_arm
