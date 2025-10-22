#include "pickee_mobile_wonho/states/error_state.hpp"
#include <chrono>

namespace pickee_mobile_wonho {

ErrorState::ErrorState(std::shared_ptr<rclcpp::Logger> logger, 
                      ErrorType error_type, 
                      const std::string& error_message)
    : logger_(logger)
    , error_type_(error_type)
    , error_message_(error_message)
    , error_start_time_(std::chrono::steady_clock::now())
    , recovery_attempts_(0)
{
    if (!logger_) {
        throw std::invalid_argument("Logger cannot be null");
    }
}

void ErrorState::OnEnter() {
    error_start_time_ = std::chrono::steady_clock::now();
    recovery_attempts_ = 0;
    
    RCLCPP_ERROR(*logger_, 
        "[ErrorState] 오류 상태에 진입했습니다. 타입: %s, 메시지: %s",
        ErrorTypeToString(error_type_).c_str(),
        error_message_.empty() ? "없음" : error_message_.c_str());
    
    // 오류 타입별 특별 처리
    switch (error_type_) {
        case ErrorType::COMMUNICATION_ERROR:
            RCLCPP_ERROR(*logger_, "[ErrorState] 통신 오류가 발생했습니다. 연결을 확인하세요.");
            break;
        case ErrorType::SENSOR_ERROR:
            RCLCPP_ERROR(*logger_, "[ErrorState] 센서 오류가 발생했습니다. 센서 상태를 확인하세요.");
            break;
        case ErrorType::ACTUATOR_ERROR:
            RCLCPP_ERROR(*logger_, "[ErrorState] 액추에이터 오류가 발생했습니다. 모터를 확인하세요.");
            break;
        case ErrorType::NAVIGATION_ERROR:
            RCLCPP_ERROR(*logger_, "[ErrorState] 네비게이션 오류가 발생했습니다.");
            break;
        case ErrorType::MEMORY_ERROR:
            RCLCPP_FATAL(*logger_, "[ErrorState] 메모리 오류가 발생했습니다. 시스템을 재시작하세요.");
            break;
        default:
            RCLCPP_ERROR(*logger_, "[ErrorState] 알 수 없는 오류가 발생했습니다.");
            break;
    }
}

void ErrorState::Execute() {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - error_start_time_);
    
    // 주기적으로 오류 상태 보고 (30초마다)
    if (elapsed.count() > 0 && elapsed.count() % 30 == 0) {
        static auto last_error_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_error_time).count() > 30000) {
            RCLCPP_ERROR(*logger_, "[ErrorState] 오류 상태 지속 중... (경과시간: %ld초, 복구시도: %d/%d, 타입: %s)",
                elapsed.count(), recovery_attempts_, MAX_RECOVERY_ATTEMPTS,
                ErrorTypeToString(error_type_).c_str());
            last_error_time = now;
        }
    }
    
    // 자동 복구 시도 (복구 가능한 경우)
    if (IsRecoverable() && recovery_attempts_ < MAX_RECOVERY_ATTEMPTS) {
        // 일정 시간 후 복구 시도 (첫 시도는 10초 후, 이후는 30초 간격)
        int wait_time = (recovery_attempts_ == 0) ? 10 : 30;
        if (elapsed.count() >= wait_time * (recovery_attempts_ + 1)) {
            RCLCPP_WARN(*logger_, "[ErrorState] 자동 복구를 시도합니다... (%d/%d)",
                recovery_attempts_ + 1, MAX_RECOVERY_ATTEMPTS);
            AttemptRecovery();
        }
    }
}

void ErrorState::OnExit() {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - error_start_time_);
    
    RCLCPP_INFO(*logger_, 
        "[ErrorState] 오류 상태를 해제합니다. "
        "(오류 지속 시간: %ld초, 복구 시도: %d회, 타입: %s)", 
        elapsed.count(), recovery_attempts_,
        ErrorTypeToString(error_type_).c_str());
}

StateType ErrorState::GetType() const {
    return StateType::ERROR;
}

ErrorState::ErrorType ErrorState::GetErrorType() const {
    return error_type_;
}

const std::string& ErrorState::GetErrorMessage() const {
    return error_message_;
}

bool ErrorState::IsRecoverable() const {
    // 메모리 오류는 복구 불가능
    if (error_type_ == ErrorType::MEMORY_ERROR) {
        return false;
    }
    
    // 최대 복구 시도 횟수에 도달하면 복구 불가능
    return recovery_attempts_ < MAX_RECOVERY_ATTEMPTS;
}

bool ErrorState::AttemptRecovery() {
    if (!IsRecoverable()) {
        RCLCPP_ERROR(*logger_, "[ErrorState] 복구가 불가능한 상태입니다.");
        return false;
    }
    
    recovery_attempts_++;
    
    RCLCPP_WARN(*logger_, "[ErrorState] 복구 시도 중... (%d/%d, 타입: %s)",
        recovery_attempts_, MAX_RECOVERY_ATTEMPTS,
        ErrorTypeToString(error_type_).c_str());
    
    // 오류 타입별 복구 로직 (현재는 시뮬레이션)
    bool recovery_success = false;
    
    switch (error_type_) {
        case ErrorType::COMMUNICATION_ERROR:
            // 통신 재연결 시도
            recovery_success = (recovery_attempts_ >= 2); // 2번째 시도에서 성공
            break;
        case ErrorType::SENSOR_ERROR:
            // 센서 재초기화 시도
            recovery_success = (recovery_attempts_ >= 1); // 1번째 시도에서 성공
            break;
        case ErrorType::ACTUATOR_ERROR:
            // 액추에이터 재시작 시도
            recovery_success = (recovery_attempts_ >= 3); // 3번째 시도에서 성공
            break;
        case ErrorType::NAVIGATION_ERROR:
            // 네비게이션 재계획 시도
            recovery_success = (recovery_attempts_ >= 2); // 2번째 시도에서 성공
            break;
        default:
            // 일반적인 재시작 시도
            recovery_success = (recovery_attempts_ >= 2); // 2번째 시도에서 성공
            break;
    }
    
    if (recovery_success) {
        RCLCPP_INFO(*logger_, "[ErrorState] 복구에 성공했습니다!");
        return true;
    } else {
        RCLCPP_WARN(*logger_, "[ErrorState] 복구 시도 실패. 재시도할 예정입니다.");
        return false;
    }
}

std::string ErrorState::ErrorTypeToString(ErrorType type) const {
    switch (type) {
        case ErrorType::COMMUNICATION_ERROR:
            return "통신 오류";
        case ErrorType::SENSOR_ERROR:
            return "센서 오류";
        case ErrorType::ACTUATOR_ERROR:
            return "액추에이터 오류";
        case ErrorType::NAVIGATION_ERROR:
            return "네비게이션 오류";
        case ErrorType::MEMORY_ERROR:
            return "메모리 오류";
        case ErrorType::UNKNOWN_ERROR:
        default:
            return "알 수 없는 오류";
    }
}

} // namespace pickee_mobile_wonho