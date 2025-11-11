#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <limits>
#include <cmath>

// Nav2 Action 헤더
// #include <nav2_msgs/action/navigate_to_pose.hpp> // 경유지 테스트를 위해 NavigateThroughPoses로 변경
#include <nav2_msgs/action/navigate_through_poses.hpp>
#include <std_srvs/srv/trigger.hpp> 

// Shopee Interface 메시지 및 서비스 헤더
#include <shopee_interfaces/msg/pickee_mobile_pose.hpp>
#include <shopee_interfaces/msg/pickee_mobile_status.hpp>
#include <shopee_interfaces/msg/pickee_mobile_arrival.hpp>
#include <shopee_interfaces/msg/pickee_mobile_speed_control.hpp>
#include <shopee_interfaces/msg/aruco_pose.hpp>
#include <shopee_interfaces/msg/person_detection.hpp>
#include <shopee_interfaces/srv/pickee_mobile_move_to_location.hpp>
#include <shopee_interfaces/srv/pickee_mobile_update_global_path.hpp>
#include <shopee_interfaces/srv/change_tracking_mode.hpp>

/**
 * @brief Shopee Pickee Mobile Controller 노드
 * 
 * Nav2와 Pickee Main Controller 간의 통신 브리지 역할을 수행합니다.
 * - Services: move_to_location, update_global_path (서버 역할)
 * - Publishers: pose, arrival
 * - Subscribers: speed_control
 */
class PickeeMobileWonhoNode : public rclcpp::Node
{
public:
    PickeeMobileWonhoNode() : Node("pickee_mobile_wonho_node")
    {
        RCLCPP_INFO(this->get_logger(), "Pickee Mobile Wonho 노드가 시작되었습니다.");
        
        // 파라미터 초기화
        this->declare_parameter<int>("robot_id", 1);
        this->declare_parameter<double>("pose_publish_rate", 10.0);
        this->declare_parameter<double>("battery_level", 100.0);
        
        robot_id_ = this->get_parameter("robot_id").as_int();
        pose_publish_rate_ = this->get_parameter("pose_publish_rate").as_double();
        battery_level_ = this->get_parameter("battery_level").as_double();
        
        // 상태 초기화
        current_order_id_ = 0;
        current_status_ = "idle";  // 'idle', 'moving', 'stopped', 'charging', 'error'
        
        // Publishers 초기화 (Shopee Interface - pub 토픽들)
        status_publisher_ = this->create_publisher<shopee_interfaces::msg::PickeeMobileStatus>(
            "/pickee/mobile/status", 10);

        pose_publisher_ = this->create_publisher<shopee_interfaces::msg::PickeeMobilePose>(
            "/pickee/mobile/pose", 10);
            
        arrival_publisher_ = this->create_publisher<shopee_interfaces::msg::PickeeMobileArrival>(
            "/pickee/mobile/arrival", 10);
        
        // custom_planner reset 서비스 클라이언트 초기화
        reset_planner_client_ = this->create_client<std_srvs::srv::Trigger>(
            "custom_planner/reset");

        // Subscribers 초기화 (Shopee Interface - sub 토픽들 + Nav2 토픽들)
        speed_control_subscriber_ = this->create_subscription<shopee_interfaces::msg::PickeeMobileSpeedControl>(
            "/pickee/mobile/speed_control", 10,
            std::bind(&PickeeMobileWonhoNode::speed_control_callback, this, std::placeholders::_1));
            
        odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&PickeeMobileWonhoNode::odom_callback, this, std::placeholders::_1));
            
        scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&PickeeMobileWonhoNode::scan_callback, this, std::placeholders::_1));

        aruco_pose_subscriber_ = this->create_subscription<shopee_interfaces::msg::ArucoPose>(
            "/pickee/mobile/aruco_pose", 10,
            std::bind(&PickeeMobileWonhoNode::aruco_pose_callback, this, std::placeholders::_1));

        person_detection_subscriber_ = this->create_subscription<shopee_interfaces::msg::PersonDetection>(
            "/pickee/mobile/person_detection", 10,
            std::bind(&PickeeMobileWonhoNode::person_detection_callback, this, std::placeholders::_1));

        // Services 초기화 (Shopee Interface - service 서버들)
        move_to_location_service_ = this->create_service<shopee_interfaces::srv::PickeeMobileMoveToLocation>(
            "/pickee/mobile/move_to_location",
            std::bind(&PickeeMobileWonhoNode::move_to_location_callback, this,
                     std::placeholders::_1, std::placeholders::_2));
                     
        update_global_path_service_ = this->create_service<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>(
            "/pickee/mobile/update_global_path",
            std::bind(&PickeeMobileWonhoNode::update_global_path_callback, this,
                     std::placeholders::_1, std::placeholders::_2));

        change_tracking_mode_service_ = this->create_service<shopee_interfaces::srv::ChangeTrackingMode>(
            "/pickee/mobile/change_tracking_mode",
            std::bind(&PickeeMobileWonhoNode::change_tracking_mode_callback, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        // Nav2 Action Client 초기화
        // nav2_action_client_ = rclcpp_action::create_client<NavigateToPose>(
        //     this, "/navigate_to_pose");
        nav2_action_client_ = rclcpp_action::create_client<NavigateThroughPoses>(
            this, "/navigate_through_poses");
            
        // Timer 초기화 - 주기적으로 위치 정보 발행
        auto timer_period = std::chrono::milliseconds(static_cast<int>(1000.0 / pose_publish_rate_));
        pose_timer_ = this->create_wall_timer(
            timer_period,
            std::bind(&PickeeMobileWonhoNode::publish_pose, this));
            
        RCLCPP_INFO(this->get_logger(), "로봇 ID: %d, 위치 발행 주기: %.1f Hz", robot_id_, pose_publish_rate_);
        RCLCPP_INFO(this->get_logger(), "Shopee 인터페이스 통신이 준비되었습니다.");
        RCLCPP_INFO(this->get_logger(), "Nav2 Action Client가 준비되었습니다.");
    }

private:
    // 노드 파라미터
    int robot_id_;
    double pose_publish_rate_;
    double battery_level_;
    
    // 현재 상태 저장
    int current_order_id_;
    std::string current_status_;
    nav_msgs::msg::Odometry current_odom_;
    sensor_msgs::msg::LaserScan current_scan_;
    bool odom_received_ = false;
    bool scan_received_ = false;
    rclcpp::Time last_scan_time_;  // 마지막 스캔 시간 추적
    
    // Publishers (Shopee Interface - pub 토픽들)
    rclcpp::Publisher<shopee_interfaces::msg::PickeeMobileStatus>::SharedPtr status_publisher_;
    rclcpp::Publisher<shopee_interfaces::msg::PickeeMobilePose>::SharedPtr pose_publisher_;
    rclcpp::Publisher<shopee_interfaces::msg::PickeeMobileArrival>::SharedPtr arrival_publisher_;
    
    // Subscribers
    rclcpp::Subscription<shopee_interfaces::msg::PickeeMobileSpeedControl>::SharedPtr speed_control_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    rclcpp::Subscription<shopee_interfaces::msg::ArucoPose>::SharedPtr aruco_pose_subscriber_;
    rclcpp::Subscription<shopee_interfaces::msg::PersonDetection>::SharedPtr person_detection_subscriber_;
    
    // Services (Shopee Interface - service 서버들)
    rclcpp::Service<shopee_interfaces::srv::PickeeMobileMoveToLocation>::SharedPtr move_to_location_service_;
    rclcpp::Service<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>::SharedPtr update_global_path_service_;
    rclcpp::Service<shopee_interfaces::srv::ChangeTrackingMode>::SharedPtr change_tracking_mode_service_;
    
    // Nav2 Action Client
    // using NavigateToPose = nav2_msgs::action::NavigateToPose;
    // rclcpp_action::Client<NavigateToPose>::SharedPtr nav2_action_client_;
    using NavigateThroughPoses = nav2_msgs::action::NavigateThroughPoses;
    rclcpp_action::Client<NavigateThroughPoses>::SharedPtr nav2_action_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_planner_client_;

    // 현재 네비게이션 상태 추적
    bool navigation_in_progress_ = false;
    int current_target_location_id_ = 0;
    shopee_interfaces::msg::Pose2D current_target_pose_;
    
    // Timers
    rclcpp::TimerBase::SharedPtr pose_timer_;
    
    /**
     * @brief 전방 장애물 감지 함수 (LiDAR 기반) - 개선 버전 with 디버깅
     * @param detection_range 감지 거리 (미터)
     * @param angle_range 감지 각도 범위 (라디안, 중심 기준 ±)
     * @return 장애물이 있으면 true, 없으면 false
     */
    bool is_obstacle_ahead(double detection_range = 0.75) // 기본값: 0.5m, ±30도
    {
        // RCLCPP_INFO(this->get_logger(),
        //     "레이저 스캔 데이터 수신됨: angle_min=%.2f, angle_max=%.2f, range_min=%.2f, range_max=%.2f, ranges.size=%zu",
        //     current_scan_->angle_min, current_scan_->angle_max, current_scan_->range_min, current_scan_->range_max, current_scan_->ranges.size());

        // 첫 5개 거리 값만 출력 (디버깅용)
        for (size_t i = 0; i < std::min(current_scan_.ranges.size(), size_t(5)); ++i) {
            if (current_scan_.ranges[i] < detection_range) {
                return true;
            } 
        }
        
        return false;
    }
    
    /**
     * @brief 주기적으로 로봇의 현재 위치와 상태를 발행합니다. (Shopee Interface)
     */
    void publish_pose()
    {
        auto pose_msg = shopee_interfaces::msg::PickeeMobilePose();
        
        pose_msg.robot_id = robot_id_;
        pose_msg.order_id = current_order_id_;
        
        if (odom_received_) {
            // Odometry에서 현재 위치 추출
            pose_msg.current_pose.x = current_odom_.pose.pose.position.x;
            pose_msg.current_pose.y = current_odom_.pose.pose.position.y;
            
            // Quaternion을 Euler angle로 변환하여 theta 계산
            auto q = current_odom_.pose.pose.orientation;
            pose_msg.current_pose.theta = atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                               1.0 - 2.0 * (q.y * q.y + q.z * q.z));
            
            // 현재 속도 정보
            pose_msg.linear_velocity = current_odom_.twist.twist.linear.x;
            pose_msg.angular_velocity = current_odom_.twist.twist.angular.z;
        } else {
            pose_msg.current_pose.x = 0.0;
            pose_msg.current_pose.y = 0.0;
            pose_msg.current_pose.theta = 0.0;
            pose_msg.linear_velocity = 0.0;
            pose_msg.angular_velocity = 0.0;
        }
        
        // 배터리 상태 확인 및 업데이트
        check_battery_status();
        
        // 배터리 잔량 (시뮬레이션)
        pose_msg.battery_level = battery_level_;
        
        // 현재 상태
        pose_msg.status = current_status_;
        
        pose_publisher_->publish(pose_msg);
        
        // 로그에도 간헐적으로 출력 (5초마다)
        static int count = 0;
        count++;
        if (count % static_cast<int>(5 * pose_publish_rate_) == 0) {
            RCLCPP_INFO(this->get_logger(), "로봇[%d] 위치: (%.2f, %.2f, %.2f) 상태: %s", 
                       robot_id_, pose_msg.current_pose.x, pose_msg.current_pose.y, 
                       pose_msg.current_pose.theta, current_status_.c_str());
        }
    }
    
    /**
     * @brief 속도 제어 명령을 처리합니다. (Shopee Interface Subscriber)
     */
    void speed_control_callback(const shopee_interfaces::msg::PickeeMobileSpeedControl::SharedPtr msg)
    {
        if (msg->robot_id != robot_id_) {
            return;  // 다른 로봇의 명령은 무시
        }
        
        RCLCPP_INFO(this->get_logger(), "속도 제어 명령 수신: 모드=%s, 속도=%.2f, 이유=%s", 
                    msg->speed_mode.c_str(), msg->target_speed, msg->reason.c_str());
        
        current_order_id_ = msg->order_id;
        
        // 상태 업데이트 ('idle', 'moving', 'stopped', 'charging', 'error')
        if (msg->speed_mode == "stop") {
            change_status("stopped", "속도 제어 명령");
        } else if (msg->speed_mode == "decelerate") {
            change_status("moving", "감속 중");
        } else if (msg->speed_mode == "normal") {
            change_status("moving", "정상 이동");
        } else {
            change_status("error", "알 수 없는 속도 모드: " + msg->speed_mode);
        }
        
        // 장애물 정보 처리
        if (!msg->obstacles.empty()) {
            RCLCPP_WARN(this->get_logger(), "장애물 %ld개 감지됨", msg->obstacles.size());
            for (const auto& obstacle : msg->obstacles) {
                RCLCPP_WARN(this->get_logger(), "  - %s: 거리 %.2f m, 속도 %.2f m/s", 
                           obstacle.obstacle_type.c_str(), obstacle.distance, obstacle.velocity);
            }
        }
        
        // TODO: Nav2에 속도 제어 명령 전달 구현
    }
    
    /**
     * @brief Odometry 콜백 - 현재 로봇의 위치와 속도 정보를 업데이트합니다.
     */
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        current_odom_ = *msg;
        odom_received_ = true;
    }
    
    /**
     * @brief LaserScan 콜백 - 레이저 스캔 데이터를 업데이트합니다.
     */
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        current_scan_ = *msg;
        scan_received_ = true;
        last_scan_time_ = this->get_clock()->now();  // 스캔 시간 업데이트
        
    }

    void aruco_pose_callback(const shopee_interfaces::msg::ArucoPose::SharedPtr msg)
    {
        geometry_msgs::msg::Twist cmd_vel_msg;
        cmd_vel_msg.linear.x = 0.0;
        cmd_vel_msg.linear.y = 0.0;
        cmd_vel_msg.linear.z = 0.0;
        cmd_vel_msg.angular.x = 0.0;
        cmd_vel_msg.angular.y = 0.0;
        cmd_vel_msg.angular.z = 0.0;
        if (current_status_ == "idle") {
            if (msg->z > 0.8) {
                // 상세 ArUco 마커 정보 출력
                RCLCPP_INFO(this->get_logger(),
                    "ArUco 마커 수신: ID=%d, 위치=(%.3f, %.3f, %.3f), 회전=(%.3f, %.3f, %.3f)",
                    msg->aruco_id,
                    msg->x, msg->y, msg->z,
                    msg->roll, msg->pitch, msg->yaw);

                // 로봇의 x 속도를 0.1로 설정하는 Twist 메시지 발행
                cmd_vel_msg.linear.x = 0.08;   // x축 속도 0.08 m/s

                // /cmd_vel 토픽으로 속도 명령 발행
                static rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub = nullptr;
                if (!cmd_vel_pub) {
                    cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
                }
                cmd_vel_pub->publish(cmd_vel_msg);
                RCLCPP_INFO(this->get_logger(), "이동 중...");
            } else {
                cmd_vel_msg.linear.x = 0.0;
                static rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub = nullptr;
                if (!cmd_vel_pub) {
                    cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
                }
                cmd_vel_pub->publish(cmd_vel_msg);
                RCLCPP_INFO(this->get_logger(), "도착");
            }
        }

    }

    void person_detection_callback(const shopee_interfaces::msg::PersonDetection::SharedPtr msg)
    {
        geometry_msgs::msg::Twist cmd_vel_msg;
        cmd_vel_msg.linear.x = 0.0;
        cmd_vel_msg.linear.y = 0.0;
        cmd_vel_msg.linear.z = 0.0;
        cmd_vel_msg.angular.x = 0.0;
        cmd_vel_msg.angular.y = 0.0;
        cmd_vel_msg.angular.z = 0.0;
        
        if (current_status_ == "tracking") {
            // 전방 장애물 감지 (1.0m 이내)
            if (is_obstacle_ahead(0.75)) {  // 1.0m
                RCLCPP_WARN(this->get_logger(), "전방에 장애물 감지! 정지합니다.");
                cmd_vel_msg.linear.x = 0.0;
                cmd_vel_msg.angular.z = 0.0;
                
                static rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub = nullptr;
                if (!cmd_vel_pub) {
                    cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
                }
                cmd_vel_pub->publish(cmd_vel_msg);
                return;  // 장애물이 있으면 이동 명령 무시
            }
            
            // 장애물이 없을 때만 이동
            if (msg->direction == "center") {
                RCLCPP_INFO(this->get_logger(),
                    "앞으로 이동중: dir=%s",
                    msg->direction.c_str());

                cmd_vel_msg.linear.x = 0.15;   // x축 속도 0.08 m/s
                
                RCLCPP_INFO(this->get_logger(), "이동 중...");
            } else if (msg->direction == "left") {
                RCLCPP_INFO(this->get_logger(),
                    "왼쪽으로 이동중: dir=%s",
                    msg->direction.c_str());

                cmd_vel_msg.linear.x = 0.15;
                cmd_vel_msg.angular.z = 0.15;   // 반시계 방향 회전

            } else if (msg->direction == "right") {
                RCLCPP_INFO(this->get_logger(),
                    "오른쪽으로 이동중: dir=%s",
                    msg->direction.c_str());

                cmd_vel_msg.linear.x = 0.15;
                cmd_vel_msg.angular.z = -0.15;   // 시계 방향 회전

            } else {
                cmd_vel_msg.linear.x = 0.0;
                RCLCPP_INFO(this->get_logger(), "도착");
            }
            
            static rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub = nullptr;
            if (!cmd_vel_pub) {
                cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
            }
            cmd_vel_pub->publish(cmd_vel_msg);
        } else if (current_status_ == "idle") {
            cmd_vel_msg.linear.x = 0.0;
            cmd_vel_msg.angular.z = 0.0;
            
            static rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub = nullptr;
            if (!cmd_vel_pub) {
                cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
            }
            cmd_vel_pub->publish(cmd_vel_msg);
            return;  // 장애물이 있으면 이동 명령 무시
        }
    }
    
    void move_to_location_callback(
        const std::shared_ptr<shopee_interfaces::srv::PickeeMobileMoveToLocation::Request> request,
        std::shared_ptr<shopee_interfaces::srv::PickeeMobileMoveToLocation::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "이동 명령 수신: 로봇=%d, 주문=%d, 위치=%d", 
                    request->robot_id, request->order_id, request->location_id);
        
        if (request->robot_id != robot_id_) {
            response->success = false;
            response->message = "잘못된 로봇 ID";
            return;
        }
        
        // Nav2 Action 서버가 사용 가능한지 확인
        RCLCPP_INFO(this->get_logger(), "Nav2 Action 서버 연결 대기 중...");
        if (!nav2_action_client_->wait_for_action_server(std::chrono::seconds(10))) {
            RCLCPP_ERROR(this->get_logger(), "Nav2 Action 서버(/navigate_through_poses)가 사용할 수 없습니다! (10초 대기 후 타임아웃)");
            response->success = false;
            response->message = "Nav2 Action 서버 연결 실패";
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Nav2 Action 서버 연결 성공!");
        
        current_order_id_ = request->order_id;
        current_target_location_id_ = request->location_id;
        current_target_pose_ = request->target_pose;
        
        RCLCPP_INFO(this->get_logger(), "목표 위치: (%.2f, %.2f, %.2f)", 
                    request->target_pose.x, request->target_pose.y, request->target_pose.theta);
        
        // Nav2 네비게이션 목표 생성 (단일 목적지)
        auto goal_msg = NavigateThroughPoses::Goal();
        goal_msg.behavior_tree = ""; // Use default behavior tree

        // 최종 목적지 생성
        geometry_msgs::msg::PoseStamped final_pose;
        final_pose.header.frame_id = "map";
        final_pose.header.stamp = this->get_clock()->now();
        final_pose.pose.position.x = request->target_pose.x;
        final_pose.pose.position.y = request->target_pose.y;
        final_pose.pose.position.z = 0.0;
        
        tf2::Quaternion q;
        q.setRPY(0, 0, request->target_pose.theta);
        final_pose.pose.orientation = tf2::toMsg(q);

        // 목표 리스트에 최종 목적지만 추가
        goal_msg.poses.push_back(final_pose);
        
        // Nav2 Action 전송 옵션 설정
        auto send_goal_options = rclcpp_action::Client<NavigateThroughPoses>::SendGoalOptions();
        
        // 목표 응답 콜백
        send_goal_options.goal_response_callback = 
            [this](rclcpp_action::ClientGoalHandle<NavigateThroughPoses>::SharedPtr goal_handle) {
                if (!goal_handle) {
                    RCLCPP_ERROR(this->get_logger(), "Nav2 네비게이션 목표가 거부되었습니다!");
                    change_status("error", "네비게이션 목표 거부");
                } else {
                    RCLCPP_INFO(this->get_logger(), "Nav2 네비게이션 목표가 수락되었습니다!");
                    navigation_in_progress_ = true;
                    change_status("moving", "Nav2 네비게이션 진행 중");
                }
            };
        
        // 피드백 콜백
        send_goal_options.feedback_callback = 
            [this](rclcpp_action::ClientGoalHandle<NavigateThroughPoses>::SharedPtr,
                   const std::shared_ptr<const NavigateThroughPoses::Feedback> feedback) {
                // 피드백이 number_of_poses_remaining 이므로, 단일 목표일때는 크게 의미 없음.
                RCLCPP_DEBUG(this->get_logger(), "Nav2 네비게이션 진행 중... 남은 경유지: %d", 
                            feedback->number_of_poses_remaining);
            };
        
        // 결과 콜백
        send_goal_options.result_callback = 
            [this](const rclcpp_action::ClientGoalHandle<NavigateThroughPoses>::WrappedResult & result) {
                navigation_in_progress_ = false;
                
                switch (result.code) {
                    case rclcpp_action::ResultCode::SUCCEEDED:
                        RCLCPP_INFO(this->get_logger(), "Nav2 네비게이션 성공! 목적지에 도착했습니다.");
                        change_status("idle", "네비게이션 완료");
                        publish_arrival_notification(current_target_location_id_, current_target_pose_);

                        // CustomPlanner 초기화 서비스 호출
                        RCLCPP_INFO(this->get_logger(), "CustomPlanner 초기화 서비스 호출 시도...");
                        if (reset_planner_client_->wait_for_service(std::chrono::seconds(5))) {
                            RCLCPP_INFO(this->get_logger(), "CustomPlanner 서비스 발견됨. 요청 전송 중...");
                            auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
                            reset_planner_client_->async_send_request(request,
                                [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
                                    try {
                                        auto response = future.get();
                                        if (response->success) {
                                            RCLCPP_INFO(this->get_logger(), "CustomPlanner 초기화 성공: %s", 
                                                response->message.c_str());
                                        } else {
                                            RCLCPP_WARN(this->get_logger(), "CustomPlanner 초기화 실패: %s", 
                                                response->message.c_str());
                                        }
                                    } catch (const std::exception& e) {
                                        RCLCPP_ERROR(this->get_logger(), "CustomPlanner 서비스 호출 예외: %s", e.what());
                                    }
                                });
                        } else {
                            RCLCPP_WARN(this->get_logger(), "CustomPlanner 초기화 서비스를 사용할 수 없습니다. (5초 대기 후 타임아웃)");
                        }
                        break;
                    case rclcpp_action::ResultCode::ABORTED:
                        RCLCPP_ERROR(this->get_logger(), "Nav2 네비게이션이 중단되었습니다.");
                        change_status("error", "네비게이션 중단");
                        break;
                    case rclcpp_action::ResultCode::CANCELED:
                        RCLCPP_WARN(this->get_logger(), "Nav2 네비게이션이 취소되었습니다.");
                        change_status("stopped", "네비게이션 취소");
                        break;
                    default:
                        RCLCPP_ERROR(this->get_logger(), "Nav2 네비게이션 결과를 알 수 없습니다.");
                        change_status("error", "네비게이션 알 수 없는 결과");
                        break;
                }
            };
        
        // Nav2 Action 전송
        RCLCPP_INFO(this->get_logger(), "Nav2에 네비게이션 목표(단일) 전송 중...");
        nav2_action_client_->async_send_goal(goal_msg, send_goal_options);
        
        response->success = true;
        response->message = "Nav2 네비게이션 시작됨";
    }
    
    /**
     * @brief 전역 경로 업데이트 명령을 처리합니다. (Shopee Interface Service)
     */
    void update_global_path_callback(
        const std::shared_ptr<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath::Request> request,
        std::shared_ptr<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "경로 업데이트 요청: 로봇=%d, 주문=%d, 위치=%d", 
                    request->robot_id, request->order_id, request->location_id);
        
        if (request->robot_id != robot_id_) {
            response->success = false;
            response->message = "잘못된 로봇 ID";
            return;
        }
        
        current_order_id_ = request->order_id;
        
        RCLCPP_INFO(this->get_logger(), "경로 업데이트 대상 위치: (%.2f, %.2f, %.2f)", 
                    request->target_pose.x, request->target_pose.y, request->target_pose.theta);
        
        // TODO: Nav2에 새 경로 전달 구현
        
        response->success = true;
        response->message = "전역 경로 업데이트됨";
    }

    void change_tracking_mode_callback(
        const std::shared_ptr<shopee_interfaces::srv::ChangeTrackingMode::Request> request,
        std::shared_ptr<shopee_interfaces::srv::ChangeTrackingMode::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "트래킹 모드 변경 요청: 로봇=%d, 모드=%s", 
                    request->robot_id, request->mode.c_str());

        if (request->robot_id != robot_id_) {
            response->success = false;
            response->message = "잘못된 로봇 ID";
            return;
        }

        response->success = true;
        response->message = "트래킹 모드 변경됨";

        if (request->mode == "idle") {
            change_status("idle", "일반 모드로 변경");
            return;
        } else if (request->mode != "tracking") {
            RCLCPP_WARN(this->get_logger(), "유효하지 않은 트래킹 모드: %s", request->mode.c_str());
            response->success = false;
            response->message = "유효하지 않은 트래킹 모드";
            return;
        } else {
            change_status("tracking", "트래킹 모드로 변경");
        }
    }
    
    /**
     * @brief 도착 알림을 발행합니다. (Shopee Interface Publisher)
     */
    void publish_arrival_notification(int location_id, const shopee_interfaces::msg::Pose2D& target_pose)
    {
        auto arrival_msg = shopee_interfaces::msg::PickeeMobileArrival();
        
        arrival_msg.robot_id = robot_id_;
        arrival_msg.order_id = current_order_id_;
        arrival_msg.location_id = location_id;
        
        if (odom_received_) {
            arrival_msg.final_pose.x = current_odom_.pose.pose.position.x;
            arrival_msg.final_pose.y = current_odom_.pose.pose.position.y;
            auto q = current_odom_.pose.pose.orientation;
            arrival_msg.final_pose.theta = atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                                1.0 - 2.0 * (q.y * q.y + q.z * q.z));
            
            // 위치 오차 계산
            arrival_msg.position_error.x = target_pose.x - arrival_msg.final_pose.x;
            arrival_msg.position_error.y = target_pose.y - arrival_msg.final_pose.y;
            arrival_msg.position_error.theta = target_pose.theta - arrival_msg.final_pose.theta;
        }
        
        // 이동 시간 계산 (시뮬레이션)
        arrival_msg.travel_time = 30.0;  // TODO: 실제 시간 계산
        arrival_msg.message = "도착 완료";
        
        arrival_publisher_->publish(arrival_msg);
        
        change_status("idle", "목적지 도착");
        
        RCLCPP_INFO(this->get_logger(), "도착 알림 발행: 위치 %d에 도착", location_id);
    }
    
    /**
     * @brief 로봇 상태를 안전하게 변경합니다.
     * @param new_status 새로운 상태 ('idle', 'moving', 'stopped', 'charging', 'error', 'tracking)
     * @param reason 상태 변경 이유
     */
    void change_status(const std::string& new_status, const std::string& reason = "")
    {
        // 유효한 상태인지 확인
        if (new_status != "idle" && new_status != "moving" && new_status != "stopped" && 
            new_status != "charging" && new_status != "error" && new_status != "tracking") {
            RCLCPP_WARN(this->get_logger(), "유효하지 않은 상태: %s", new_status.c_str());
            current_status_ = "error";
            return;
        }
        
        if (current_status_ != new_status) {
            RCLCPP_INFO(this->get_logger(), "로봇 상태 변경: %s -> %s %s", 
                       current_status_.c_str(), new_status.c_str(), 
                       reason.empty() ? "" : ("(" + reason + ")").c_str());
            current_status_ = new_status;

            shopee_interfaces::msg::PickeeMobileStatus status_msg;
            status_msg.robot_id = robot_id_;
            status_msg.status = current_status_;

            status_publisher_->publish(status_msg);
        }
    }
    
    /**
     * @brief 배터리 레벨에 따른 상태 확인 및 업데이트
     */
    void check_battery_status()
    {
        if (battery_level_ < 20.0 && current_status_ != "charging") {
            change_status("charging", "배터리 부족");
        } else if (battery_level_ >= 80.0 && current_status_ == "charging") {
            change_status("idle", "충전 완료");
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<PickeeMobileWonhoNode>();
    
    RCLCPP_INFO(node->get_logger(), "Pickee Mobile Wonho 서비스 실행 중...");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}
