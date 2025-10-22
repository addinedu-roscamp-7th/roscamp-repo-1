#pragma once

#include <memory>
#include <atomic>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

namespace pickee_mobile_wonho {

class MotionControlComponent {
public:
    explicit MotionControlComponent(std::shared_ptr<rclcpp::Logger> logger);
    ~MotionControlComponent() = default;

    geometry_msgs::msg::Twist ComputeControlCommand();
    void EmergencyStop();
    void SetSpeedLimits(double max_linear, double max_angular);

private:
    std::shared_ptr<rclcpp::Logger> logger_;
    std::atomic<bool> emergency_stop_;
    double max_linear_speed_;
    double max_angular_speed_;
};

} // namespace pickee_mobile_wonho