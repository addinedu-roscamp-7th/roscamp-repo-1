#include "pickee_mobile_wonho/mobile_controller.hpp"
#include "pickee_mobile_wonho/states/idle_state.hpp"
#include "pickee_mobile_wonho/states/error_state.hpp"

namespace pickee_mobile_wonho {

PickeeMobileController::PickeeMobileController(const rclcpp::NodeOptions& options)
    : Node("pickee_mobile_controller", options)
    , robot_id_(1)
    , default_linear_speed_(0.5)
    , default_angular_speed_(0.3)
    , battery_threshold_low_(20.0)
    , arrival_position_tolerance_(0.05)
    , arrival_angle_tolerance_(0.1)
    , path_planning_frequency_(10.0)
    , motion_control_frequency_(20.0)
{
    RCLCPP_INFO(this->get_logger(), "=== Pickee Mobile Controller 시작 ===");
    
    try {
        // 초기화 단계별 실행
        SafeExecute([this]() { DeclareParameters(); }, "파라미터 선언");
        SafeExecute([this]() { LoadParameters(); }, "파라미터 로딩");
        SafeExecute([this]() { InitializeComponents(); }, "컴포넌트 초기화");
        SafeExecute([this]() { InitializeRosInterfaces(); }, "ROS2 인터페이스 초기화");
        SafeExecute([this]() { InitializeStateMachine(); }, "상태 기계 초기화");
        
        RCLCPP_INFO(this->get_logger(), "Pickee Mobile Controller 초기화 완료!");
        
    } catch (const std::exception& e) {
        RCLCPP_FATAL(this->get_logger(), "초기화 실패: %s", e.what());
        throw;
    }
}

void PickeeMobileController::SafeExecute(std::function<void()> func, const std::string& operation_name) {
    try {
        func();
        RCLCPP_DEBUG(this->get_logger(), "%s 완료", operation_name.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "%s 중 오류 발생: %s", operation_name.c_str(), e.what());
        TransitionToErrorState("초기화 오류: " + operation_name);
        throw;
    }
}

void PickeeMobileController::DeclareParameters() {
    this->declare_parameter("robot_id", robot_id_);
    this->declare_parameter("default_linear_speed", default_linear_speed_);
    this->declare_parameter("default_angular_speed", default_angular_speed_);
    this->declare_parameter("battery_threshold_low", battery_threshold_low_);
    this->declare_parameter("arrival_position_tolerance", arrival_position_tolerance_);
    this->declare_parameter("arrival_angle_tolerance", arrival_angle_tolerance_);
    this->declare_parameter("path_planning_frequency", path_planning_frequency_);
    this->declare_parameter("motion_control_frequency", motion_control_frequency_);
}

void PickeeMobileController::LoadParameters() {
    robot_id_ = this->get_parameter("robot_id").as_int();
    default_linear_speed_ = this->get_parameter("default_linear_speed").as_double();
    default_angular_speed_ = this->get_parameter("default_angular_speed").as_double();
    battery_threshold_low_ = this->get_parameter("battery_threshold_low").as_double();
    arrival_position_tolerance_ = this->get_parameter("arrival_position_tolerance").as_double();
    arrival_angle_tolerance_ = this->get_parameter("arrival_angle_tolerance").as_double();
    path_planning_frequency_ = this->get_parameter("path_planning_frequency").as_double();
    motion_control_frequency_ = this->get_parameter("motion_control_frequency").as_double();
    
    RCLCPP_INFO(this->get_logger(), 
        "파라미터 로딩 완료 - Robot ID: %d, 선속도: %.2f, 각속도: %.2f",
        robot_id_, default_linear_speed_, default_angular_speed_);
}

void PickeeMobileController::InitializeComponents() {
    auto logger = std::make_shared<rclcpp::Logger>(this->get_logger());
    
    // 핵심 컴포넌트 초기화
    localization_component_ = std::make_shared<LocalizationComponent>(logger);
    path_planning_component_ = std::make_shared<PathPlanningComponent>(logger);
    motion_control_component_ = std::make_shared<MotionControlComponent>(logger);
    sensor_manager_ = std::make_shared<SensorManager>(logger);
    battery_manager_ = std::make_shared<BatteryManager>(logger);
    arrival_detector_ = std::make_shared<ArrivalDetector>(logger);
    communication_interface_ = std::make_shared<CommunicationInterface>(logger);
    
    RCLCPP_INFO(this->get_logger(), "모든 컴포넌트가 초기화되었습니다.");
}

void PickeeMobileController::InitializeRosInterfaces() {
    // Transform 관리 초기화
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    
    // 퍼블리셔 초기화
    cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    battery_status_publisher_ = this->create_publisher<std_msgs::msg::Float64>("/battery_status", 10);
    
    // 구독자 초기화
    laser_scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10, std::bind(&PickeeMobileController::LaserScanCallback, this, std::placeholders::_1));
    
    imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", 10, std::bind(&PickeeMobileController::ImuCallback, this, std::placeholders::_1));
    
    odometry_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10, std::bind(&PickeeMobileController::OdometryCallback, this, std::placeholders::_1));
    
    // 타이머 초기화
    main_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / motion_control_frequency_)),
        std::bind(&PickeeMobileController::MainTimerCallback, this));
    
    pose_report_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),  // 100ms = 10Hz
        std::bind(&PickeeMobileController::PoseReportTimerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), "ROS2 인터페이스가 초기화되었습니다.");
}

void PickeeMobileController::InitializeStateMachine() {
    auto logger = std::make_shared<rclcpp::Logger>(this->get_logger());
    
    // 상태 기계 생성
    state_machine_ = std::make_unique<StateMachine>(logger);
    
    // 상태 전환 콜백 설정
    state_machine_->SetStateTransitionCallback(
        std::bind(&PickeeMobileController::StateTransitionCallback, this, 
                  std::placeholders::_1, std::placeholders::_2));
    
    // 초기 상태로 IDLE 설정
    auto idle_state = std::make_unique<IdleState>(logger);
    state_machine_->TransitionTo(std::move(idle_state));
    
    RCLCPP_INFO(this->get_logger(), "상태 기계가 IDLE 상태로 초기화되었습니다.");
}

void PickeeMobileController::MainTimerCallback() {
    SafeExecute([this]() {
        if (state_machine_) {
            state_machine_->Execute();
        }
    }, "상태 기계 실행");
}

void PickeeMobileController::PoseReportTimerCallback() {
    SafeExecute([this]() {
        if (localization_component_ && battery_manager_) {
            // 위치 및 배터리 상태 보고 (향후 구현)
            RCLCPP_DEBUG(this->get_logger(), "위치 및 배터리 상태 보고 중...");
        }
    }, "위치 보고");
}

void PickeeMobileController::LaserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    SafeExecute([this, msg]() {
        if (localization_component_) {
            localization_component_->UpdateSensorData(msg);
        }
    }, "레이저 스캔 처리");
}

void PickeeMobileController::ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    SafeExecute([this, msg]() {
        if (localization_component_) {
            localization_component_->UpdateImu(msg);
        }
    }, "IMU 데이터 처리");
}

void PickeeMobileController::OdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    SafeExecute([this, msg]() {
        if (localization_component_) {
            localization_component_->UpdateOdometry(msg);
        }
    }, "오도메트리 데이터 처리");
}

void PickeeMobileController::StateTransitionCallback(StateType old_state, StateType new_state) {
    RCLCPP_INFO(this->get_logger(), 
        "상태 전환: %d -> %d", 
        static_cast<int>(old_state), static_cast<int>(new_state));
}

void PickeeMobileController::TransitionToErrorState(const std::string& error_message) {
    if (!state_machine_) return;
    
    try {
        auto logger = std::make_shared<rclcpp::Logger>(this->get_logger());
        auto error_state = std::make_unique<ErrorState>(
            logger, ErrorState::ErrorType::UNKNOWN_ERROR, error_message);
        state_machine_->TransitionTo(std::move(error_state));
    } catch (...) {
        RCLCPP_FATAL(this->get_logger(), "ERROR 상태로 전환 실패!");
    }
}

} // namespace pickee_mobile_wonho