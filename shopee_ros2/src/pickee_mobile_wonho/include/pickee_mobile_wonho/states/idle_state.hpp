#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/states/state.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief 대기 상태 클래스
 * 
 * Pickee Main Controller로부터 이동 명령을 대기하는 상태입니다.
 */
class IdleState : public State {
public:
    /**
     * @brief 생성자
     * @param logger ROS2 로거 포인터
     */
    explicit IdleState(std::shared_ptr<rclcpp::Logger> logger);

    /**
     * @brief 소멸자
     */
    ~IdleState() override = default;

    /**
     * @brief 상태 진입 시 호출
     */
    void OnEnter() override;

    /**
     * @brief 상태 실행 중 주기적으로 호출
     */
    void Execute() override;

    /**
     * @brief 상태 이탈 시 호출
     */
    void OnExit() override;

    /**
     * @brief 상태 타입 반환
     * @return StateType::IDLE
     */
    StateType GetType() const override;

private:
    /**
     * @brief ROS2 로거
     */
    std::shared_ptr<rclcpp::Logger> logger_;

    /**
     * @brief 대기 시작 시간
     */
    std::chrono::steady_clock::time_point idle_start_time_;
};

} // namespace pickee_mobile_wonho