#pragma once

#include <memory>
#include <functional>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/states/state.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief 상태 기계 클래스
 * 
 * Pickee Mobile의 상태를 관리하고 안전한 상태 전환을 제공합니다.
 * 스마트 포인터를 활용하여 메모리 안전성을 보장합니다.
 */
class StateMachine {
public:
    /**
     * @brief 상태 전환 콜백 함수 타입
     */
    using StateTransitionCallback = std::function<void(StateType, StateType)>;

    /**
     * @brief 생성자
     * @param logger ROS2 로거 포인터
     */
    explicit StateMachine(std::shared_ptr<rclcpp::Logger> logger);

    /**
     * @brief 소멸자
     */
    ~StateMachine() = default;

    /**
     * @brief 새로운 상태로 전환
     * @param new_state 새로운 상태 (unique_ptr)
     * @throw std::runtime_error 상태 전환 실패 시
     */
    void TransitionTo(std::unique_ptr<State> new_state);

    /**
     * @brief 현재 상태 실행
     * 
     * 현재 상태의 Execute() 메서드를 호출합니다.
     * 예외가 발생할 경우 ERROR 상태로 전환됩니다.
     */
    void Execute();

    /**
     * @brief 현재 상태 타입 반환
     * @return StateType 현재 상태 타입
     */
    StateType GetCurrentStateType() const;

    /**
     * @brief 상태 전환 콜백 설정
     * @param callback 상태 전환 시 호출될 콜백 함수
     */
    void SetStateTransitionCallback(StateTransitionCallback callback);

    /**
     * @brief 상태 기계가 초기화되었는지 확인
     * @return true 초기화됨, false 초기화되지 않음
     */
    bool IsInitialized() const;

private:
    /**
     * @brief 현재 상태 (스마트 포인터로 관리)
     */
    std::unique_ptr<State> current_state_;

    /**
     * @brief ROS2 로거
     */
    std::shared_ptr<rclcpp::Logger> logger_;

    /**
     * @brief 상태 전환 콜백 함수
     */
    StateTransitionCallback transition_callback_;

    /**
     * @brief 이전 상태 타입 (로깅용)
     */
    StateType previous_state_type_;

    /**
     * @brief 복사 생성자 삭제
     */
    StateMachine(const StateMachine&) = delete;

    /**
     * @brief 할당 연산자 삭제
     */
    StateMachine& operator=(const StateMachine&) = delete;
};

} // namespace pickee_mobile_wonho