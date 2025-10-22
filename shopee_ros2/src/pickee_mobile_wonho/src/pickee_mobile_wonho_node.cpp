#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Shopee Interface 메시지 및 서비스 헤더
#include <shopee_interfaces/msg/pickee_mobile_pose.hpp>
#include <shopee_interfaces/msg/pickee_mobile_arrival.hpp>
#include <shopee_interfaces/msg/pickee_mobile_speed_control.hpp>
#include <shopee_interfaces/srv/pickee_mobile_move_to_location.hpp>
#include <shopee_interfaces/srv/pickee_mobile_update_global_path.hpp>

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
        pose_publisher_ = this->create_publisher<shopee_interfaces::msg::PickeeMobilePose>(
            "/pickee/mobile/pose", 10);
            
        arrival_publisher_ = this->create_publisher<shopee_interfaces::msg::PickeeMobileArrival>(
            "/pickee/mobile/arrival", 10);
            
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
        
        // Services 초기화 (Shopee Interface - service 서버들)
        move_to_location_service_ = this->create_service<shopee_interfaces::srv::PickeeMobileMoveToLocation>(
            "/pickee/mobile/move_to_location",
            std::bind(&PickeeMobileWonhoNode::move_to_location_callback, this,
                     std::placeholders::_1, std::placeholders::_2));
                     
        update_global_path_service_ = this->create_service<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>(
            "/pickee/mobile/update_global_path",
            std::bind(&PickeeMobileWonhoNode::update_global_path_callback, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        // Timer 초기화 - 주기적으로 위치 정보 발행
        auto timer_period = std::chrono::milliseconds(static_cast<int>(1000.0 / pose_publish_rate_));
        pose_timer_ = this->create_wall_timer(
            timer_period,
            std::bind(&PickeeMobileWonhoNode::publish_pose, this));
            
        RCLCPP_INFO(this->get_logger(), "로봇 ID: %d, 위치 발행 주기: %.1f Hz", robot_id_, pose_publish_rate_);
        RCLCPP_INFO(this->get_logger(), "Shopee 인터페이스 통신이 준비되었습니다.");
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
    
    // Publishers (Shopee Interface - pub 토픽들)
    rclcpp::Publisher<shopee_interfaces::msg::PickeeMobilePose>::SharedPtr pose_publisher_;
    rclcpp::Publisher<shopee_interfaces::msg::PickeeMobileArrival>::SharedPtr arrival_publisher_;
    
    // Subscribers
    rclcpp::Subscription<shopee_interfaces::msg::PickeeMobileSpeedControl>::SharedPtr speed_control_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    
    // Services (Shopee Interface - service 서버들)
    rclcpp::Service<shopee_interfaces::srv::PickeeMobileMoveToLocation>::SharedPtr move_to_location_service_;
    rclcpp::Service<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>::SharedPtr update_global_path_service_;
    
    // Timers
    rclcpp::TimerBase::SharedPtr pose_timer_;
    
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
    }
    
    /**
     * @brief 목적지 이동 명령을 처리합니다. (Shopee Interface Service)
     */
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
        
        current_order_id_ = request->order_id;
        change_status("moving", "네비게이션 시작");
        
        RCLCPP_INFO(this->get_logger(), "목표 위치: (%.2f, %.2f, %.2f)", 
                    request->target_pose.x, request->target_pose.y, request->target_pose.theta);
        
        // TODO: Nav2에 네비게이션 목표 전달 구현
        
        response->success = true;
        response->message = "네비게이션 시작됨";
        
        // 잠시 후 도착 알림 시뮬레이션 (실제로는 Nav2 완료 콜백에서 처리)
        // publish_arrival_notification(request->location_id, request->target_pose);
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
     * @param new_status 새로운 상태 ('idle', 'moving', 'stopped', 'charging', 'error')
     * @param reason 상태 변경 이유
     */
    void change_status(const std::string& new_status, const std::string& reason = "")
    {
        // 유효한 상태인지 확인
        if (new_status != "idle" && new_status != "moving" && new_status != "stopped" && 
            new_status != "charging" && new_status != "error") {
            RCLCPP_WARN(this->get_logger(), "유효하지 않은 상태: %s", new_status.c_str());
            current_status_ = "error";
            return;
        }
        
        if (current_status_ != new_status) {
            RCLCPP_INFO(this->get_logger(), "로봇 상태 변경: %s -> %s %s", 
                       current_status_.c_str(), new_status.c_str(), 
                       reason.empty() ? "" : ("(" + reason + ")").c_str());
            current_status_ = new_status;
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
