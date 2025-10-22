#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "pickee_mobile_wonho/states/state.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief 이동 상태 클래스
 * 
 * 지정된 목적지로 이동하는 상태입니다.
 */
class MovingState : public State {
public:
    /**
     * @brief 생성자
     * @param logger ROS2 로거 포인터
     */
    explicit MovingState(std::shared_ptr<rclcpp::Logger> logger);

    /**
     * @brief 소멸자
     */
    ~MovingState() override = default;

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
     * @return StateType::MOVING
     */
    StateType GetType() const override;

    /**
     * @brief 목적지 설정
     * @param target_pose 목표 위치
     */
    void SetTargetPose(const geometry_msgs::msg::PoseStamped& target_pose);

    /**
     * @brief 이동 진행률 반환
     * @return 진행률 (0.0 ~ 1.0)
     */
    double GetProgress() const;

private:
    /**
     * @brief ROS2 로거
     */
    std::shared_ptr<rclcpp::Logger> logger_;

    /**
     * @brief 목표 위치
     */
    geometry_msgs::msg::PoseStamped target_pose_;

    /**
     * @brief 이동 시작 시간
     */
    std::chrono::steady_clock::time_point move_start_time_;

    /**
     * @brief 목표가 설정되었는지 여부
     */
    bool target_set_;

    /**
     * @brief 이동 진행률
     */
    double progress_;
};

} // namespace pickee_mobile_wonho