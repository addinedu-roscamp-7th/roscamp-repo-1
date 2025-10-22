#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/states/state.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief 충전 상태 클래스
 * 
 * 배터리 충전 중인 상태입니다.
 */
class ChargingState : public State {
public:
    /**
     * @brief 생성자
     * @param logger ROS2 로거 포인터
     */
    explicit ChargingState(std::shared_ptr<rclcpp::Logger> logger);

    /**
     * @brief 소멸자
     */
    ~ChargingState() override = default;

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
     * @return StateType::CHARGING
     */
    StateType GetType() const override;

    /**
     * @brief 현재 배터리 레벨 설정
     * @param level 배터리 레벨 (0.0 ~ 1.0)
     */
    void SetBatteryLevel(double level);

    /**
     * @brief 현재 배터리 레벨 반환
     * @return 배터리 레벨 (0.0 ~ 1.0)
     */
    double GetBatteryLevel() const;

    /**
     * @brief 충전 완료 여부 확인
     * @return true 충전 완료, false 충전 중
     */
    bool IsChargingComplete() const;

private:
    /**
     * @brief ROS2 로거
     */
    std::shared_ptr<rclcpp::Logger> logger_;

    /**
     * @brief 충전 시작 시간
     */
    std::chrono::steady_clock::time_point charge_start_time_;

    /**
     * @brief 현재 배터리 레벨 (0.0 ~ 1.0)
     */
    double battery_level_;

    /**
     * @brief 충전 시작 시 배터리 레벨
     */
    double initial_battery_level_;

    /**
     * @brief 충전 완료 임계값
     */
    static constexpr double CHARGE_COMPLETE_THRESHOLD = 0.95; // 95%

    /**
     * @brief 충전 속도 시뮬레이션 (1분당 10% 충전)
     */
    static constexpr double CHARGE_RATE_PER_MINUTE = 0.10;
};

} // namespace pickee_mobile_wonho