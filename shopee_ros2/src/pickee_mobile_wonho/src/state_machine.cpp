#include "pickee_mobile_wonho/state_machine.hpp"
#include "pickee_mobile_wonho/states/error_state.hpp"

namespace pickee_mobile_wonho {

StateMachine::StateMachine(std::shared_ptr<rclcpp::Logger> logger)
    : current_state_(nullptr)
    , logger_(logger)
    , previous_state_type_(StateType::IDLE)
{
    if (!logger_) {
        throw std::invalid_argument("Logger cannot be null");
    }
    
    RCLCPP_INFO(*logger_, "상태 기계가 초기화되었습니다.");
}

void StateMachine::TransitionTo(std::unique_ptr<State> new_state) {
    if (!new_state) {
        RCLCPP_ERROR(*logger_, "새로운 상태가 null입니다.");
        throw std::invalid_argument("New state cannot be null");
    }

    try {
        // 이전 상태 타입 저장
        StateType old_state_type = current_state_ ? 
            current_state_->GetType() : StateType::IDLE;
        StateType new_state_type = new_state->GetType();

        // 현재 상태 종료
        if (current_state_) {
            RCLCPP_INFO(*logger_, "상태 이탈: %d", static_cast<int>(old_state_type));
            current_state_->OnExit();
        }

        // 새로운 상태로 전환
        current_state_ = std::move(new_state);
        
        // 새로운 상태 진입
        RCLCPP_INFO(*logger_, "상태 진입: %d", static_cast<int>(new_state_type));
        current_state_->OnEnter();

        // 상태 전환 콜백 호출
        if (transition_callback_) {
            transition_callback_(old_state_type, new_state_type);
        }

        previous_state_type_ = old_state_type;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(*logger_, "상태 전환 중 오류 발생: %s", e.what());
        // 오류 발생 시 ERROR 상태로 전환 시도
        // (무한 재귀 방지를 위해 현재 상태가 ERROR가 아닐 때만)
        if (current_state_ && current_state_->GetType() != StateType::ERROR) {
            try {
                current_state_ = std::make_unique<ErrorState>(logger_);
                current_state_->OnEnter();
            } catch (...) {
                RCLCPP_FATAL(*logger_, "ERROR 상태로 전환 실패. 시스템을 중단합니다.");
                current_state_.reset();
            }
        }
        throw;
    }
}

void StateMachine::Execute() {
    if (!current_state_) {
        RCLCPP_WARN(*logger_, "실행할 상태가 없습니다.");
        return;
    }

    try {
        current_state_->Execute();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(*logger_, "상태 실행 중 오류 발생: %s", e.what());
        
        // ERROR 상태로 전환 (현재 상태가 ERROR가 아닐 때만)
        if (current_state_->GetType() != StateType::ERROR) {
            try {
                auto error_state = std::make_unique<ErrorState>(logger_);
                TransitionTo(std::move(error_state));
            } catch (...) {
                RCLCPP_FATAL(*logger_, "ERROR 상태로 전환 실패. 시스템을 중단합니다.");
                current_state_.reset();
            }
        }
    }
}

StateType StateMachine::GetCurrentStateType() const {
    return current_state_ ? current_state_->GetType() : StateType::IDLE;
}

void StateMachine::SetStateTransitionCallback(StateTransitionCallback callback) {
    transition_callback_ = callback;
}

bool StateMachine::IsInitialized() const {
    return current_state_ != nullptr;
}

} // namespace pickee_mobile_wonho