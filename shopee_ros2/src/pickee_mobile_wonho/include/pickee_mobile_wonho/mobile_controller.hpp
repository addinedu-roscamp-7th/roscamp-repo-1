#pragma once

#include <memory>
#include <string>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float64.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

// Shopee 인터페이스 (미래에 구현될 예정)
// #include "shopee_interfaces/srv/pickee_mobile_move_to_location.hpp"
// #include "shopee_interfaces/srv/pickee_mobile_update_global_path.hpp"
// #include "shopee_interfaces/msg/pickee_mobile_pose.hpp"
// #include "shopee_interfaces/msg/pickee_mobile_arrival.hpp"
// #include "shopee_interfaces/msg/pickee_mobile_speed_control.hpp"

#include "pickee_mobile_wonho/state_machine.hpp"
#include "pickee_mobile_wonho/components/localization_component.hpp"
#include "pickee_mobile_wonho/components/path_planning_component.hpp"
#include "pickee_mobile_wonho/components/motion_control_component.hpp"
#include "pickee_mobile_wonho/components/sensor_manager.hpp"
#include "pickee_mobile_wonho/components/battery_manager.hpp"
#include "pickee_mobile_wonho/components/arrival_detector.hpp"
#include "pickee_mobile_wonho/communication_interface.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief Pickee Mobile 메인 컨트롤러 클래스
 * 
 * ROS2 노드로서 Pickee Mobile 로봇의 모든 기능을 통합 관리합니다.
 * Modern C++17 및 스마트 포인터를 활용한 안전하고 효율적인 구현을 제공합니다.
 */
class PickeeMobileController : public rclcpp::Node {
public:
    /**
     * @brief 생성자
     * @param options 노드 옵션
     */
    explicit PickeeMobileController(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    /**
     * @brief 소멸자
     */
    ~PickeeMobileController() override = default;

    /**
     * @brief 안전한 실행 메서드
     * @param func 실행할 함수
     * @param operation_name 작업 이름 (로깅용)
     */
    void SafeExecute(std::function<void()> func, const std::string& operation_name);

private:
    // === 초기화 메서드들 ===
    
    /**
     * @brief 파라미터 선언
     */
    void DeclareParameters();

    /**
     * @brief 파라미터 로딩
     */
    void LoadParameters();

    /**
     * @brief 컴포넌트 초기화
     */
    void InitializeComponents();

    /**
     * @brief ROS2 인터페이스 초기화
     */
    void InitializeRosInterfaces();

    /**
     * @brief 상태 기계 초기화
     */
    void InitializeStateMachine();

    // === 콜백 메서드들 ===

    /**
     * @brief 메인 타이머 콜백 (상태 실행)
     */
    void MainTimerCallback();

    /**
     * @brief 위치 보고 타이머 콜백
     */
    void PoseReportTimerCallback();

    /**
     * @brief 센서 데이터 콜백들
     */
    void LaserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void OdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

    /**
     * @brief 상태 전환 콜백
     * @param old_state 이전 상태
     * @param new_state 새로운 상태
     */
    void StateTransitionCallback(StateType old_state, StateType new_state);

    /**
     * @brief ERROR 상태로 전환
     * @param error_message 오류 메시지
     */
    void TransitionToErrorState(const std::string& error_message = "");

    // === 멤버 변수들 ===

    // 파라미터들
    int robot_id_;
    double default_linear_speed_;
    double default_angular_speed_;
    double battery_threshold_low_;
    double arrival_position_tolerance_;
    double arrival_angle_tolerance_;
    double path_planning_frequency_;
    double motion_control_frequency_;

    // 상태 기계
    std::unique_ptr<StateMachine> state_machine_;

    // 핵심 컴포넌트들
    std::shared_ptr<LocalizationComponent> localization_component_;
    std::shared_ptr<PathPlanningComponent> path_planning_component_;
    std::shared_ptr<MotionControlComponent> motion_control_component_;
    std::shared_ptr<SensorManager> sensor_manager_;
    std::shared_ptr<BatteryManager> battery_manager_;
    std::shared_ptr<ArrivalDetector> arrival_detector_;

    // 통신 인터페이스
    std::shared_ptr<CommunicationInterface> communication_interface_;

    // ROS2 퍼블리셔들
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr battery_status_publisher_;

    // ROS2 구독자들
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;

    // Transform 관리
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    // 타이머들
    rclcpp::TimerBase::SharedPtr main_timer_;
    rclcpp::TimerBase::SharedPtr pose_report_timer_;

    // 삭제된 생성자들 (복사 방지)
    PickeeMobileController(const PickeeMobileController&) = delete;
    PickeeMobileController& operator=(const PickeeMobileController&) = delete;
};

} // namespace pickee_mobile_wonho