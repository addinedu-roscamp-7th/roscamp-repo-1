#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/states/state.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief 정지 상태 클래스
 * 
 * 장애물이나 명령에 의해 이동이 중단된 상태입니다.
 */
class StoppedState : public State {
public:
    /**
     * @brief 정지 사유 열거형
     */
    enum class StopReason {
        OBSTACLE_DETECTED,      ///< 장애물 감지
        EMERGENCY_STOP,         ///< 비상 정지
        USER_COMMAND,           ///< 사용자 명령
        SYSTEM_ERROR,           ///< 시스템 오류
        UNKNOWN                 ///< 알 수 없는 이유
    };

    /**
     * @brief 생성자
     * @param logger ROS2 로거 포인터
     * @param reason 정지 사유
     */
    explicit StoppedState(std::shared_ptr<rclcpp::Logger> logger, 
                         StopReason reason = StopReason::UNKNOWN);

    /**
     * @brief 소멸자
     */
    ~StoppedState() override = default;

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
     * @return StateType::STOPPED
     */
    StateType GetType() const override;

    /**
     * @brief 정지 사유 반환
     * @return StopReason 정지 사유
     */
    StopReason GetStopReason() const;

    /**
     * @brief 재시작 준비가 되었는지 확인
     * @return true 재시작 가능, false 재시작 불가
     */
    bool IsReadyToResume() const;

private:
    /**
     * @brief ROS2 로거
     */
    std::shared_ptr<rclcpp::Logger> logger_;

    /**
     * @brief 정지 사유
     */
    StopReason stop_reason_;

    /**
     * @brief 정지 시작 시간
     */
    std::chrono::steady_clock::time_point stop_start_time_;

    /**
     * @brief 정지 사유를 문자열로 변환
     * @param reason 정지 사유
     * @return 사유 문자열
     */
    std::string StopReasonToString(StopReason reason) const;
};

} // namespace pickee_mobile_wonho