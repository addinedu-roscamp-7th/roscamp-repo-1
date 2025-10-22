#include "pickee_mobile_wonho/states/moving_state.hpp"

namespace pickee_mobile_wonho {

MovingState::MovingState(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
    , move_start_time_(std::chrono::steady_clock::now())
    , target_set_(false)
    , progress_(0.0)
{
    if (!logger_) {
        throw std::invalid_argument("Logger cannot be null");
    }
}

void MovingState::OnEnter() {
    move_start_time_ = std::chrono::steady_clock::now();
    progress_ = 0.0;
    RCLCPP_INFO(*logger_, "[MovingState] 이동 상태에 진입했습니다.");
    
    if (!target_set_) {
        RCLCPP_WARN(*logger_, "[MovingState] 목표 위치가 설정되지 않았습니다.");
    }
}

void MovingState::Execute() {
    if (!target_set_) {
        static auto last_warn_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_warn_time).count() > 5000) {
            RCLCPP_WARN(*logger_, "[MovingState] 목표 위치가 설정되지 않아 이동할 수 없습니다.");
            last_warn_time = now;
        }
        return;
    }

    // 이동 로직 실행 (현재는 시뮬레이션)
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - move_start_time_);
    
    // 임시로 10초 후 완료로 시뮬레이션
    progress_ = std::min(1.0, elapsed.count() / 10.0);
    
    RCLCPP_DEBUG(*logger_, 
        "[MovingState] 이동 중... 진행률: %.1f%% (%.2f, %.2f, %.2f)", 
        progress_ * 100.0,
        target_pose_.pose.position.x,
        target_pose_.pose.position.y,
        target_pose_.pose.position.z);

    if (progress_ >= 1.0) {
        RCLCPP_INFO(*logger_, "[MovingState] 목적지에 도착했습니다!");
    }
}

void MovingState::OnExit() {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - move_start_time_);
    
    RCLCPP_INFO(*logger_, 
        "[MovingState] 이동 상태를 종료합니다. (이동 시간: %ldms, 진행률: %.1f%%)", 
        elapsed.count(), progress_ * 100.0);
}

StateType MovingState::GetType() const {
    return StateType::MOVING;
}

void MovingState::SetTargetPose(const geometry_msgs::msg::PoseStamped& target_pose) {
    target_pose_ = target_pose;
    target_set_ = true;
    
    RCLCPP_INFO(*logger_, 
        "[MovingState] 목표 위치 설정: (%.2f, %.2f, %.2f)",
        target_pose.pose.position.x,
        target_pose.pose.position.y,
        target_pose.pose.position.z);
}

double MovingState::GetProgress() const {
    return progress_;
}

} // namespace pickee_mobile_wonho