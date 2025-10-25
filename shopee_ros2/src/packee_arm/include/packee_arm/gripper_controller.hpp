#ifndef PACKEE_ARM_GRIPPER_CONTROLLER_HPP_
#define PACKEE_ARM_GRIPPER_CONTROLLER_HPP_

#include <mutex>
#include <string>

#include "rclcpp/node.hpp"
#include "rclcpp/logger.hpp"
#include "std_msgs/msg/float32.hpp"

namespace packee_arm {

// GripperController 클래스는 그리퍼 힘 제어를 담당한다.
class GripperController {
public:
  GripperController(
    rclcpp::Node * node,
    const rclcpp::Logger & logger,
    double force_limit,
    const std::string & left_gripper_topic,
    const std::string & right_gripper_topic);

  bool Close(const std::string & arm_side, double requested_force);

  bool Open(const std::string & arm_side);

  void UpdateForceLimit(double force_limit);

  double GetForceLimit() const;

private:
  void PublishGripperCommand(
    const std::string & arm_side,
    double force_newton);

  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr GetPublisher(const std::string & arm_side);

  rclcpp::Node * node_;
  rclcpp::Logger logger_;
  mutable std::mutex mutex_;
  double force_limit_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr left_gripper_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr right_gripper_pub_;
};

}  // namespace packee_arm

#endif  // PACKEE_ARM_GRIPPER_CONTROLLER_HPP_
