#ifndef PICKEE_MOBILE_WONHO_BRINGUP_HPP
#define PICKEE_MOBILE_WONHO_BRINGUP_HPP

#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "pickee_mobile_wonho/zlac_driver.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief Pickee Mobile 로봇 하드웨어 제어 노드
 * 
 * ZLAC 모터 드라이버를 통해 실제 로봇 하드웨어를 제어하고,
 * Nav2와 연동하기 위한 odometry, TF, joint state 등을 발행합니다.
 */
class BringupNode : public rclcpp::Node {
public:
    /**
     * @brief 생성자
     */
    BringupNode();

    /**
     * @brief 소멸자
     */
    ~BringupNode();

private:
    // 로봇 물리적 파라미터
    struct RobotParams {
        double wheel_radius;        // 바퀴 반지름 (m)
        double wheel_base;          // 바퀴 간격 (m)
        double gear_ratio;          // 기어비
        int encoder_resolution;     // 엔코더 해상도 (pulse/rev)
        double max_linear_vel;      // 최대 선속도 (m/s)
        double max_angular_vel;     // 최대 각속도 (rad/s)
        double cmd_vel_timeout;     // cmd_vel 타임아웃 (s)
        std::string base_frame;     // 베이스 프레임 ID
        std::string odom_frame;     // odometry 프레임 ID
    };

    // 오도메트리 상태
    struct OdometryState {
        double x;           // X 위치 (m)
        double y;           // Y 위치 (m)
        double theta;       // 회전각 (rad)
        double v_x;         // X 방향 속도 (m/s)
        double v_y;         // Y 방향 속도 (m/s)
        double v_theta;     // 각속도 (rad/s)
        rclcpp::Time last_update;   // 마지막 업데이트 시간
    };

    // 모터 상태
    struct MotorState {
        int32_t left_position;      // 좌측 모터 위치 (pulse)
        int32_t right_position;     // 우측 모터 위치 (pulse)
        int32_t left_velocity;      // 좌측 모터 속도 (RPM)
        int32_t right_velocity;     // 우측 모터 속도 (RPM)
        int32_t prev_left_position; // 이전 좌측 모터 위치
        int32_t prev_right_position;// 이전 우측 모터 위치
        bool left_enabled;          // 좌측 모터 활성화 상태
        bool right_enabled;         // 우측 모터 활성화 상태
    };

    // ROS 파라미터 초기화
    void InitializeParameters();

    // 하드웨어 초기화
    bool InitializeHardware();

    // 타이머 콜백 함수들
    void ControlTimerCallback();
    void StatusTimerCallback();

    // cmd_vel 콜백
    void CmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);

    // 속도 명령을 모터 RPM으로 변환
    void ConvertVelocityToMotorRPM(double linear_vel, double angular_vel, 
                                   int32_t& left_rpm, int32_t& right_rpm);

    // 모터 위치에서 오도메트리 계산
    void UpdateOdometry();

    // 오도메트리 발행
    void PublishOdometry();

    // TF 발행
    void PublishTransform();

    // Joint State 발행
    void PublishJointState();

    // 모터 상태 업데이트
    bool UpdateMotorState();

    // 비상정지
    void EmergencyStop();

    // 안전 체크
    bool SafetyCheck();

    // ZLAC 모터 드라이버
    std::unique_ptr<ZlacDriver> motor_driver_;

    // 로봇 파라미터
    RobotParams robot_params_;

    // 현재 상태
    OdometryState odom_state_;
    MotorState motor_state_;

    // 목표 속도
    double target_linear_vel_;
    double target_angular_vel_;
    rclcpp::Time last_cmd_vel_time_;

    // ROS 퍼블리셔
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;

    // ROS 구독자
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    // TF 브로드캐스터
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // 타이머
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr status_timer_;

    // 제어 주파수
    double control_frequency_;
    double status_frequency_;

    // 시리얼 포트 설정
    std::string serial_device_;
    int serial_baudrate_;
    int serial_timeout_;

    // 상태 플래그
    bool hardware_initialized_;
    bool motors_enabled_;
    bool emergency_stop_active_;

    // 로그 관련
    rclcpp::Time last_log_time_;
    double log_frequency_;

    /**
     * @brief 파라미터 유효성 검사
     * @return 유효한 경우 true
     */
    bool ValidateParameters();

    /**
     * @brief 각도 정규화 (-π ~ π)
     * @param angle 입력 각도 (rad)
     * @return 정규화된 각도 (rad)
     */
    double NormalizeAngle(double angle);

    /**
     * @brief RPM을 rad/s로 변환
     * @param rpm RPM 값
     * @return rad/s 값
     */
    double RPMToRadPerSec(int32_t rpm);

    /**
     * @brief rad/s를 RPM으로 변환
     * @param rad_per_sec rad/s 값
     * @return RPM 값
     */
    int32_t RadPerSecToRPM(double rad_per_sec);

    /**
     * @brief 펄스를 라디안으로 변환
     * @param pulse 펄스 값
     * @return 라디안 값
     */
    double PulseToRadian(int32_t pulse);

    /**
     * @brief 라디안을 펄스로 변환
     * @param radian 라디안 값
     * @return 펄스 값
     */
    int32_t RadianToPulse(double radian);
};

} // namespace pickee_mobile_wonho

#endif // PICKEE_MOBILE_WONHO_BRINGUP_HPP