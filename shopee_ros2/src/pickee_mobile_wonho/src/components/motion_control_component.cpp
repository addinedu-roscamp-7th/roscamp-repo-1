#include "pickee_mobile_wonho/components/motion_control_component.hpp"

namespace pickee_mobile_wonho {

MotionControlComponent::MotionControlComponent(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
    , emergency_stop_(false)
    , max_linear_speed_(0.5)
    , max_angular_speed_(0.3)
{
    RCLCPP_INFO(*logger_, "[MotionControlComponent] 모션 제어 컴포넌트 초기화 완료");
}

geometry_msgs::msg::Twist MotionControlComponent::ComputeControlCommand() {
    geometry_msgs::msg::Twist cmd_vel;
    
    if (emergency_stop_.load()) {
        // 비상 정지 상태
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = 0.0;
        RCLCPP_WARN_THROTTLE(*logger_, *rclcpp::Clock().get_clock_type_rcl_clock_t(), 1000,
            "[MotionControlComponent] 비상 정지 활성화됨");
    } else {
        // TODO: PID 제어 구현
        // 임시로 기본 속도 반환
        cmd_vel.linear.x = 0.0;  // 현재는 정지
        cmd_vel.angular.z = 0.0;
    }
    
    return cmd_vel;
}

void MotionControlComponent::EmergencyStop() {
    emergency_stop_.store(true);
    RCLCPP_ERROR(*logger_, "[MotionControlComponent] 비상 정지 활성화!");
}

void MotionControlComponent::SetSpeedLimits(double max_linear, double max_angular) {
    max_linear_speed_ = max_linear;
    max_angular_speed_ = max_angular;
    
    RCLCPP_INFO(*logger_, "[MotionControlComponent] 속도 제한 설정: 선속도=%.2f, 각속도=%.2f",
        max_linear_speed_, max_angular_speed_);
}

} // namespace pickee_mobile_wonho