#include "pickee_mobile_wonho/states/stopped_state.hpp"
#include <chrono>

namespace pickee_mobile_wonho {

StoppedState::StoppedState(std::shared_ptr<rclcpp::Logger> logger, StopReason reason)
    : logger_(logger)
    , stop_reason_(reason)
    , stop_start_time_(std::chrono::steady_clock::now())
{
    if (!logger_) {
        throw std::invalid_argument("Logger cannot be null");
    }
}

void StoppedState::OnEnter() {
    stop_start_time_ = std::chrono::steady_clock::now();
    
    RCLCPP_WARN(*logger_, "[StoppedState] 정지 상태에 진입했습니다. 사유: %s",
        StopReasonToString(stop_reason_).c_str());
    
    // 정지 사유에 따른 추가 처리
    switch (stop_reason_) {
        case StopReason::EMERGENCY_STOP:
            RCLCPP_ERROR(*logger_, "[StoppedState] 비상 정지가 활성화되었습니다!");
            break;
        case StopReason::OBSTACLE_DETECTED:
            RCLCPP_WARN(*logger_, "[StoppedState] 장애물이 감지되어 정지합니다.");
            break;
        case StopReason::SYSTEM_ERROR:
            RCLCPP_ERROR(*logger_, "[StoppedState] 시스템 오류로 인해 정지합니다.");
            break;
        default:
            break;
    }
}

void StoppedState::Execute() {
    // 주기적으로 정지 상태임을 확인 (10초마다)
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stop_start_time_);
    
    if (elapsed.count() > 0 && elapsed.count() % 10 == 0) {
        static auto last_warn_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_warn_time).count() > 10000) {
            RCLCPP_WARN(*logger_, "[StoppedState] 정지 상태 유지 중... (정지 시간: %ld초, 사유: %s)",
                elapsed.count(), StopReasonToString(stop_reason_).c_str());
            last_warn_time = now;
        }
    }

    // 비상 정지의 경우 추가 모니터링
    if (stop_reason_ == StopReason::EMERGENCY_STOP) {
        static auto last_error_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_error_time).count() > 5000) {
            RCLCPP_ERROR(*logger_, "[StoppedState] 비상 정지 상태입니다. 수동 해제가 필요합니다.");
            last_error_time = now;
        }
    }
}

void StoppedState::OnExit() {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stop_start_time_);
    
    RCLCPP_INFO(*logger_, 
        "[StoppedState] 정지 상태를 해제합니다. (정지 시간: %ld초, 사유: %s)", 
        elapsed.count(), StopReasonToString(stop_reason_).c_str());
}

StateType StoppedState::GetType() const {
    return StateType::STOPPED;
}

StoppedState::StopReason StoppedState::GetStopReason() const {
    return stop_reason_;
}

bool StoppedState::IsReadyToResume() const {
    // 비상 정지와 시스템 오류는 수동 해제 필요
    if (stop_reason_ == StopReason::EMERGENCY_STOP || 
        stop_reason_ == StopReason::SYSTEM_ERROR) {
        return false;
    }
    
    // 기타 사유는 일정 시간 후 재시작 가능
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stop_start_time_);
    
    return elapsed.count() >= 2; // 2초 후 재시작 가능
}

std::string StoppedState::StopReasonToString(StopReason reason) const {
    switch (reason) {
        case StopReason::OBSTACLE_DETECTED:
            return "장애물 감지";
        case StopReason::EMERGENCY_STOP:
            return "비상 정지";
        case StopReason::USER_COMMAND:
            return "사용자 명령";
        case StopReason::SYSTEM_ERROR:
            return "시스템 오류";
        case StopReason::UNKNOWN:
        default:
            return "알 수 없음";
    }
}

} // namespace pickee_mobile_wonho