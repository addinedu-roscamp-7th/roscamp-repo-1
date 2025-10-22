#include "pickee_mobile_wonho/states/idle_state.hpp"

namespace pickee_mobile_wonho {

IdleState::IdleState(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
    , idle_start_time_(std::chrono::steady_clock::now())
{
    if (!logger_) {
        throw std::invalid_argument("Logger cannot be null");
    }
}

void IdleState::OnEnter() {
    idle_start_time_ = std::chrono::steady_clock::now();
    RCLCPP_INFO(*logger_, "[IdleState] 대기 상태에 진입했습니다.");
}

void IdleState::Execute() {
    // 주기적으로 대기 상태임을 알림 (30초마다)
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        current_time - idle_start_time_);
    
    if (elapsed.count() > 0 && elapsed.count() % 30 == 0) {
        RCLCPP_DEBUG(*logger_, 
            "[IdleState] 명령 대기 중... (경과 시간: %ld초)", elapsed.count());
    }
}

void IdleState::OnExit() {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - idle_start_time_);
    
    RCLCPP_INFO(*logger_, 
        "[IdleState] 대기 상태를 종료합니다. (대기 시간: %ld초)", elapsed.count());
}

StateType IdleState::GetType() const {
    return StateType::IDLE;
}

} // namespace pickee_mobile_wonho