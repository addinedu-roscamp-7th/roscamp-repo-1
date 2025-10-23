#include "pickee_mobile_wonho/bringup.hpp"
#include <cmath>
#include <chrono>

namespace pickee_mobile_wonho {

BringupNode::BringupNode() 
    : Node("pickee_mobile_bringup")
    , target_linear_vel_(0.0)
    , target_angular_vel_(0.0)
    , hardware_initialized_(false)
    , motors_enabled_(false)
    , emergency_stop_active_(false)
{
    RCLCPP_INFO(get_logger(), "Pickee Mobile 하드웨어 노드 초기화 시작");

    // 파라미터 초기화
    InitializeParameters();

    // 오도메트리 상태 초기화
    odom_state_.x = 0.0;
    odom_state_.y = 0.0;
    odom_state_.theta = 0.0;
    odom_state_.v_x = 0.0;
    odom_state_.v_y = 0.0;
    odom_state_.v_theta = 0.0;
    odom_state_.last_update = this->now();

    // 모터 상태 초기화
    memset(&motor_state_, 0, sizeof(motor_state_));

    // ROS 퍼블리셔 생성
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odom", 50);
    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);

    // ROS 구독자 생성
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10, 
        std::bind(&BringupNode::CmdVelCallback, this, std::placeholders::_1)
    );

    // TF 브로드캐스터 생성
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // 하드웨어 초기화
    if (!InitializeHardware()) {
        RCLCPP_FATAL(get_logger(), "하드웨어 초기화 실패!");
        return;
    }

    // 타이머 생성
    control_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / control_frequency_),
        std::bind(&BringupNode::ControlTimerCallback, this)
    );

    status_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / status_frequency_),
        std::bind(&BringupNode::StatusTimerCallback, this)
    );

    last_cmd_vel_time_ = this->now();
    last_log_time_ = this->now();

    RCLCPP_INFO(get_logger(), "Pickee Mobile 하드웨어 노드 초기화 완료");
}

BringupNode::~BringupNode() {
    if (motor_driver_) {
        // 모든 모터 정지
        motor_driver_->SetVelocity(1, 0);
        motor_driver_->SetVelocity(2, 0);
        
        // 모터 비활성화
        motor_driver_->DisableMotor(1);
        motor_driver_->DisableMotor(2);
    }
    
    RCLCPP_INFO(get_logger(), "Pickee Mobile 하드웨어 노드 종료");
}

void BringupNode::InitializeParameters() {
    // 로봇 물리적 파라미터
    declare_parameter("wheel_radius", 0.08);              // 8cm 바퀴
    declare_parameter("wheel_base", 0.46);                // 46cm 바퀴 간격
    declare_parameter("gear_ratio", 50.0);                // 기어비 50:1
    declare_parameter("encoder_resolution", 1000);        // 엔코더 해상도
    declare_parameter("max_linear_vel", 2.0);             // 최대 선속도 2m/s
    declare_parameter("max_angular_vel", 3.14);           // 최대 각속도 π rad/s
    declare_parameter("cmd_vel_timeout", 1.0);            // cmd_vel 타임아웃 1초

    // 프레임 ID
    declare_parameter("base_frame", "base_footprint");
    declare_parameter("odom_frame", "odom");

    // 제어 주파수
    declare_parameter("control_frequency", 50.0);         // 제어 주파수 50Hz
    declare_parameter("status_frequency", 10.0);          // 상태 발행 주파수 10Hz

    // 시리얼 포트 설정
    declare_parameter("serial_device", "/dev/ttyUSB0");
    declare_parameter("serial_baudrate", 115200);
    declare_parameter("serial_timeout", 100);

    // 로그 주파수
    declare_parameter("log_frequency", 1.0);              // 1Hz

    // 파라미터 읽기
    robot_params_.wheel_radius = get_parameter("wheel_radius").as_double();
    robot_params_.wheel_base = get_parameter("wheel_base").as_double();
    robot_params_.gear_ratio = get_parameter("gear_ratio").as_double();
    robot_params_.encoder_resolution = get_parameter("encoder_resolution").as_int();
    robot_params_.max_linear_vel = get_parameter("max_linear_vel").as_double();
    robot_params_.max_angular_vel = get_parameter("max_angular_vel").as_double();
    robot_params_.cmd_vel_timeout = get_parameter("cmd_vel_timeout").as_double();
    robot_params_.base_frame = get_parameter("base_frame").as_string();
    robot_params_.odom_frame = get_parameter("odom_frame").as_string();

    control_frequency_ = get_parameter("control_frequency").as_double();
    status_frequency_ = get_parameter("status_frequency").as_double();

    serial_device_ = get_parameter("serial_device").as_string();
    serial_baudrate_ = get_parameter("serial_baudrate").as_int();
    serial_timeout_ = get_parameter("serial_timeout").as_int();

    log_frequency_ = get_parameter("log_frequency").as_double();

    // 파라미터 유효성 검사
    if (!ValidateParameters()) {
        RCLCPP_FATAL(get_logger(), "파라미터 유효성 검사 실패!");
        throw std::runtime_error("Invalid parameters");
    }

    RCLCPP_INFO(get_logger(), "파라미터 초기화 완료");
    RCLCPP_INFO(get_logger(), "바퀴 반지름: %.3f m, 바퀴 간격: %.3f m", 
                robot_params_.wheel_radius, robot_params_.wheel_base);
    RCLCPP_INFO(get_logger(), "시리얼 포트: %s, 속도: %d", 
                serial_device_.c_str(), serial_baudrate_);
}

bool BringupNode::InitializeHardware() {
    // ZLAC 모터 드라이버 생성 및 초기화
    motor_driver_ = std::make_unique<ZlacDriver>(serial_device_, serial_baudrate_, serial_timeout_);
    
    if (!motor_driver_->Initialize()) {
        RCLCPP_ERROR(get_logger(), "ZLAC 드라이버 초기화 실패");
        return false;
    }

    // 안전을 위해 먼저 모든 모터 정지
    motor_driver_->EmergencyStop(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 알람 클리어
    motor_driver_->ClearAlarm(1);
    motor_driver_->ClearAlarm(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 속도 제어 모드로 설정
    if (!motor_driver_->SetControlMode(1, ZlacDriver::ControlMode::VELOCITY_MODE) ||
        !motor_driver_->SetControlMode(2, ZlacDriver::ControlMode::VELOCITY_MODE)) {
        RCLCPP_ERROR(get_logger(), "모터 제어 모드 설정 실패");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 모터 활성화
    if (!motor_driver_->EnableMotor(1) || !motor_driver_->EnableMotor(2)) {
        RCLCPP_ERROR(get_logger(), "모터 활성화 실패");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 초기 위치 읽기
    motor_driver_->ReadPosition(1, motor_state_.left_position);
    motor_driver_->ReadPosition(2, motor_state_.right_position);
    motor_state_.prev_left_position = motor_state_.left_position;
    motor_state_.prev_right_position = motor_state_.right_position;

    hardware_initialized_ = true;
    motors_enabled_ = true;
    emergency_stop_active_ = false;

    RCLCPP_INFO(get_logger(), "하드웨어 초기화 완료");
    return true;
}

void BringupNode::ControlTimerCallback() {
    if (!hardware_initialized_) {
        return;
    }

    // 안전 체크
    if (!SafetyCheck()) {
        EmergencyStop();
        return;
    }

    // cmd_vel 타임아웃 체크
    auto current_time = this->now();
    double time_since_last_cmd = (current_time - last_cmd_vel_time_).seconds();
    
    if (time_since_last_cmd > robot_params_.cmd_vel_timeout) {
        // 타임아웃 시 정지
        target_linear_vel_ = 0.0;
        target_angular_vel_ = 0.0;
    }

    // 속도를 모터 RPM으로 변환
    int32_t left_rpm, right_rpm;
    ConvertVelocityToMotorRPM(target_linear_vel_, target_angular_vel_, left_rpm, right_rpm);

    // 모터에 속도 명령 전송
    if (motors_enabled_ && !emergency_stop_active_) {
        motor_driver_->SetVelocity(1, left_rpm);   // 좌측 모터
        motor_driver_->SetVelocity(2, right_rpm);  // 우측 모터
    }

    // 모터 상태 업데이트 및 오도메트리 계산
    if (UpdateMotorState()) {
        UpdateOdometry();
    }
}

void BringupNode::StatusTimerCallback() {
    if (!hardware_initialized_) {
        return;
    }

    // 오도메트리 발행
    PublishOdometry();
    
    // TF 발행
    PublishTransform();
    
    // Joint State 발행
    PublishJointState();

    // 주기적 로그 출력
    auto current_time = this->now();
    double time_since_last_log = (current_time - last_log_time_).seconds();
    
    if (time_since_last_log >= 1.0 / log_frequency_) {
        RCLCPP_INFO(get_logger(), 
                   "위치: (%.3f, %.3f, %.3f), 속도: (%.3f, %.3f), 모터: (%d, %d) RPM",
                   odom_state_.x, odom_state_.y, odom_state_.theta,
                   target_linear_vel_, target_angular_vel_,
                   motor_state_.left_velocity, motor_state_.right_velocity);
        last_log_time_ = current_time;
    }
}

void BringupNode::CmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    if (!motors_enabled_ || emergency_stop_active_) {
        return;
    }

    // 속도 제한
    target_linear_vel_ = std::max(-robot_params_.max_linear_vel, 
                                 std::min(robot_params_.max_linear_vel, msg->linear.x));
    target_angular_vel_ = std::max(-robot_params_.max_angular_vel,
                                  std::min(robot_params_.max_angular_vel, msg->angular.z));

    last_cmd_vel_time_ = this->now();
}

void BringupNode::ConvertVelocityToMotorRPM(double linear_vel, double angular_vel,
                                           int32_t& left_rpm, int32_t& right_rpm) {
    // 차동 구동 로봇의 바퀴 속도 계산
    // v_left = linear_vel - (angular_vel * wheel_base / 2)
    // v_right = linear_vel + (angular_vel * wheel_base / 2)
    
    double left_wheel_vel = linear_vel - (angular_vel * robot_params_.wheel_base / 2.0);
    double right_wheel_vel = linear_vel + (angular_vel * robot_params_.wheel_base / 2.0);

    // 바퀴 각속도를 RPM으로 변환
    // angular_velocity = linear_velocity / radius
    // RPM = (angular_velocity * 60) / (2 * π) * gear_ratio
    
    double left_angular_vel = left_wheel_vel / robot_params_.wheel_radius;
    double right_angular_vel = right_wheel_vel / robot_params_.wheel_radius;
    
    left_rpm = static_cast<int32_t>(RadPerSecToRPM(left_angular_vel));
    right_rpm = static_cast<int32_t>(RadPerSecToRPM(right_angular_vel));
}

void BringupNode::UpdateOdometry() {
    auto current_time = this->now();
    double dt = (current_time - odom_state_.last_update).seconds();
    
    if (dt <= 0.0) {
        return;
    }

    // 모터 위치 변화량 계산 (펄스)
    int32_t left_delta = motor_state_.left_position - motor_state_.prev_left_position;
    int32_t right_delta = motor_state_.right_position - motor_state_.prev_right_position;

    // 펄스를 라디안으로 변환
    double left_wheel_delta = PulseToRadian(left_delta);
    double right_wheel_delta = PulseToRadian(right_delta);

    // 바퀴 이동 거리 계산
    double left_distance = left_wheel_delta * robot_params_.wheel_radius;
    double right_distance = right_wheel_delta * robot_params_.wheel_radius;

    // 로봇 중심점 이동 거리 및 회전각 계산
    double delta_distance = (left_distance + right_distance) / 2.0;
    double delta_theta = (right_distance - left_distance) / robot_params_.wheel_base;

    // 오도메트리 업데이트
    double delta_x = delta_distance * cos(odom_state_.theta + delta_theta / 2.0);
    double delta_y = delta_distance * sin(odom_state_.theta + delta_theta / 2.0);

    odom_state_.x += delta_x;
    odom_state_.y += delta_y;
    odom_state_.theta = NormalizeAngle(odom_state_.theta + delta_theta);

    // 속도 계산
    odom_state_.v_x = delta_distance / dt;
    odom_state_.v_y = 0.0; // 차동구동 로봇은 측면 이동 없음
    odom_state_.v_theta = delta_theta / dt;

    // 이전 위치 업데이트
    motor_state_.prev_left_position = motor_state_.left_position;
    motor_state_.prev_right_position = motor_state_.right_position;
    odom_state_.last_update = current_time;
}

void BringupNode::PublishOdometry() {
    auto odom_msg = nav_msgs::msg::Odometry();
    
    odom_msg.header.stamp = this->now();
    odom_msg.header.frame_id = robot_params_.odom_frame;
    odom_msg.child_frame_id = robot_params_.base_frame;

    // 위치 정보
    odom_msg.pose.pose.position.x = odom_state_.x;
    odom_msg.pose.pose.position.y = odom_state_.y;
    odom_msg.pose.pose.position.z = 0.0;

    // 쿼터니언으로 변환
    tf2::Quaternion q;
    q.setRPY(0, 0, odom_state_.theta);
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();

    // 속도 정보
    odom_msg.twist.twist.linear.x = odom_state_.v_x;
    odom_msg.twist.twist.linear.y = odom_state_.v_y;
    odom_msg.twist.twist.angular.z = odom_state_.v_theta;

    // 공분산 행렬 (임시로 고정값 사용)
    std::fill(odom_msg.pose.covariance.begin(), odom_msg.pose.covariance.end(), 0.0);
    odom_msg.pose.covariance[0] = 0.001;  // x
    odom_msg.pose.covariance[7] = 0.001;  // y
    odom_msg.pose.covariance[35] = 0.001; // theta

    std::fill(odom_msg.twist.covariance.begin(), odom_msg.twist.covariance.end(), 0.0);
    odom_msg.twist.covariance[0] = 0.001;  // vx
    odom_msg.twist.covariance[35] = 0.001; // vtheta

    odom_pub_->publish(odom_msg);
}

void BringupNode::PublishTransform() {
    geometry_msgs::msg::TransformStamped transform;
    
    transform.header.stamp = this->now();
    transform.header.frame_id = robot_params_.odom_frame;
    transform.child_frame_id = robot_params_.base_frame;

    transform.transform.translation.x = odom_state_.x;
    transform.transform.translation.y = odom_state_.y;
    transform.transform.translation.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, odom_state_.theta);
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    transform.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(transform);
}

void BringupNode::PublishJointState() {
    auto joint_msg = sensor_msgs::msg::JointState();
    
    joint_msg.header.stamp = this->now();
    joint_msg.name = {"left_wheel_joint", "right_wheel_joint"};
    
    // 현재 바퀴 위치 (라디안)
    double left_position = PulseToRadian(motor_state_.left_position);
    double right_position = PulseToRadian(motor_state_.right_position);
    
    joint_msg.position = {left_position, right_position};
    
    // 현재 바퀴 속도 (rad/s)
    double left_velocity = RPMToRadPerSec(motor_state_.left_velocity);
    double right_velocity = RPMToRadPerSec(motor_state_.right_velocity);
    
    joint_msg.velocity = {left_velocity, right_velocity};
    
    joint_state_pub_->publish(joint_msg);
}

bool BringupNode::UpdateMotorState() {
    if (!motor_driver_) {
        return false;
    }

    // 모터 위치 및 속도 읽기
    bool success = true;
    success &= motor_driver_->ReadPosition(1, motor_state_.left_position);
    success &= motor_driver_->ReadPosition(2, motor_state_.right_position);
    success &= motor_driver_->ReadVelocity(1, motor_state_.left_velocity);
    success &= motor_driver_->ReadVelocity(2, motor_state_.right_velocity);

    if (!success) {
        RCLCPP_WARN(get_logger(), "모터 상태 읽기 실패");
        return false;
    }

    return true;
}

void BringupNode::EmergencyStop() {
    if (motor_driver_) {
        motor_driver_->EmergencyStop(0);
        target_linear_vel_ = 0.0;
        target_angular_vel_ = 0.0;
        emergency_stop_active_ = true;
        RCLCPP_ERROR(get_logger(), "비상정지 활성화!");
    }
}

bool BringupNode::SafetyCheck() {
    if (!motor_driver_ || !hardware_initialized_) {
        return false;
    }

    // 모터 상태 체크
    ZlacDriver::MotorStatus left_status, right_status;
    if (!motor_driver_->ReadMotorStatus(1, left_status) || 
        !motor_driver_->ReadMotorStatus(2, right_status)) {
        RCLCPP_WARN(get_logger(), "모터 상태 읽기 실패");
        return false;
    }

    // 알람 체크
    if (left_status.is_alarm || right_status.is_alarm) {
        RCLCPP_ERROR(get_logger(), "모터 알람 발생! 좌측: %d, 우측: %d", 
                    left_status.is_alarm, right_status.is_alarm);
        return false;
    }

    // 모터 활성화 상태 체크
    if (!left_status.is_enabled || !right_status.is_enabled) {
        RCLCPP_WARN(get_logger(), "모터 비활성화 상태! 좌측: %d, 우측: %d",
                   left_status.is_enabled, right_status.is_enabled);
        return false;
    }

    return true;
}

bool BringupNode::ValidateParameters() {
    if (robot_params_.wheel_radius <= 0.0 || robot_params_.wheel_base <= 0.0) {
        RCLCPP_ERROR(get_logger(), "잘못된 바퀴 파라미터");
        return false;
    }
    
    if (robot_params_.gear_ratio <= 0.0 || robot_params_.encoder_resolution <= 0) {
        RCLCPP_ERROR(get_logger(), "잘못된 모터 파라미터");
        return false;
    }
    
    if (control_frequency_ <= 0.0 || status_frequency_ <= 0.0) {
        RCLCPP_ERROR(get_logger(), "잘못된 주파수 설정");
        return false;
    }
    
    return true;
}

double BringupNode::NormalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double BringupNode::RPMToRadPerSec(int32_t rpm) {
    return (static_cast<double>(rpm) * 2.0 * M_PI) / (60.0 * robot_params_.gear_ratio);
}

int32_t BringupNode::RadPerSecToRPM(double rad_per_sec) {
    return static_cast<int32_t>((rad_per_sec * 60.0 * robot_params_.gear_ratio) / (2.0 * M_PI));
}

double BringupNode::PulseToRadian(int32_t pulse) {
    return (static_cast<double>(pulse) * 2.0 * M_PI) / 
           (robot_params_.encoder_resolution * robot_params_.gear_ratio);
}

int32_t BringupNode::RadianToPulse(double radian) {
    return static_cast<int32_t>((radian * robot_params_.encoder_resolution * robot_params_.gear_ratio) / 
                               (2.0 * M_PI));
}

} // namespace pickee_mobile_wonho

// main 함수
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<pickee_mobile_wonho::BringupNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("pickee_mobile_bringup"), 
                    "노드 실행 중 오류 발생: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}