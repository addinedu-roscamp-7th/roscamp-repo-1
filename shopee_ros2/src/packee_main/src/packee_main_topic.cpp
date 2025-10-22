#include "rclcpp/rclcpp.hpp"
#include "shopee_interfaces/msg/packee_packing_complete.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"
#include "shopee_interfaces/msg/packee_availability.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/arm_task_status.hpp"

using namespace std::placeholders;
using PackingComplete = shopee_interfaces::msg::PackeePackingComplete;
using RobotStatus = shopee_interfaces::msg::PackeeRobotStatus;
using Availability = shopee_interfaces::msg::PackeeAvailability;
using PoseStatus = shopee_interfaces::msg::ArmPoseStatus;
using ArmTaskStatus = shopee_interfaces::msg::ArmTaskStatus;

class PackingTopicServer : public rclcpp::Node
{
public:
    PackingTopicServer()
    : Node("packing_topic_server")
    {
        // Subscribers 생성
        pose_status_sub_ = this->create_subscription<PoseStatus>(
            "/packee/arm/pose_status",
            10,
            std::bind(&PackingTopicServer::PoseStatusCallback, this, _1)
        );

        pick_status_sub_ = this->create_subscription<ArmTaskStatus>(
            "/packee/arm/pick_status",
            10,
            std::bind(&PackingTopicServer::PickStatusCallback, this, _1)
        );

        place_status_sub_ = this->create_subscription<ArmTaskStatus>(
            "/packee/arm/place_status",
            10,
            std::bind(&PackingTopicServer::PlaceStatusCallback, this, _1)
        );

        // Publishers 생성
        packing_complete_pub_ = this->create_publisher<PackingComplete>(
            "/packee/packing_complete", 10);

        robot_status_pub_ = this->create_publisher<RobotStatus>(
            "/packee/robot_status", 10);

        availability_result_pub_ = this->create_publisher<Availability>(
            "/packee/availability_result", 10);

        // 테스트용 타이머 (5초마다 메시지 발행)
        timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&PackingTopicServer::publishTestMessages, this));

        RCLCPP_INFO(this->get_logger(), "Packing Topic Server started");
    }

    // Subscribers
    void PoseStatusCallback(const PoseStatus::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), 
                    "PoseStatus - robot_id: %d, order_id: %d, pose_type: %s, status: %s, progress: %.2f", 
                    msg->robot_id, msg->order_id, msg->pose_type.c_str(), 
                    msg->status.c_str(), msg->progress);
    }

    void PickStatusCallback(const ArmTaskStatus::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), 
                    "PickStatus - robot_id: %d, order_id: %d, product_id: %d, arm_side: %s, status: %s, current_phase: %s, progress: %.2f, message: %s", 
                    msg->robot_id, msg->order_id, msg->product_id, msg->arm_side.c_str(), 
                    msg->status.c_str(), msg->current_phase.c_str(), msg->progress, msg->message.c_str());
    }

    void PlaceStatusCallback(const ArmTaskStatus::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), 
                    "PlaceStatus - robot_id: %d, order_id: %d, product_id: %d, arm_side: %s, status: %s, current_phase: %s, progress: %.2f, message: %s", 
                    msg->robot_id, msg->order_id, msg->product_id, msg->arm_side.c_str(), 
                    msg->status.c_str(), msg->current_phase.c_str(), msg->progress, msg->message.c_str());
    }

    // Publisher
    void publishPackingComplete(int32_t robot_id, int32_t order_id, 
                                bool success, int32_t packed_items, 
                                const std::string& message)
    {
        auto msg = PackingComplete();
        msg.robot_id = robot_id;
        msg.order_id = order_id;
        msg.success = success;
        msg.packed_items = packed_items;
        msg.message = message;

        packing_complete_pub_->publish(msg);
        
        RCLCPP_INFO(this->get_logger(), 
            "Published PackingComplete: robot_id=%d, order_id=%d, success=%s, packed_items=%d",
            robot_id, order_id, success ? "true" : "false", packed_items);
    }

    void publishRobotStatus(int32_t robot_id, const std::string& state,
                           int32_t current_order_id, int32_t items_in_cart)
    {
        auto msg = RobotStatus();
        msg.robot_id = robot_id;
        msg.state = state;  // "idle", "packing", "moving", etc.
        msg.current_order_id = current_order_id;
        msg.items_in_cart = items_in_cart;

        robot_status_pub_->publish(msg);
        
        RCLCPP_INFO(this->get_logger(), 
            "Published RobotStatus: robot_id=%d, state=%s, order_id=%d, items=%d",
            robot_id, state.c_str(), current_order_id, items_in_cart);
    }

    void publishAvailabilityResult(int32_t robot_id, int32_t order_id,
                                   bool available, bool cart_detected,
                                   const std::string& message)
    {
        auto msg = Availability();
        msg.robot_id = robot_id;
        msg.order_id = order_id;
        msg.available = available;
        msg.cart_detected = cart_detected;
        msg.message = message;

        availability_result_pub_->publish(msg);
        
        RCLCPP_INFO(this->get_logger(), 
            "Published AvailabilityResult: robot_id=%d, order_id=%d, available=%s, cart=%s",
            robot_id, order_id, available ? "true" : "false", 
            cart_detected ? "true" : "false");
    }

private:
    void publishTestMessages()
    {
        static int counter = 0;
        counter++;

        if (counter % 3 == 1) {
            // 성공 케이스
            publishPackingComplete(1, 3, true, 5, "Packing completed");
            publishRobotStatus(1, "packing", 3, 5);
            publishAvailabilityResult(1, 3, true, true, "Ready for packing");
        }
        else if (counter % 3 == 2) {
            // 실패 케이스
            publishPackingComplete(1, 3, false, 3, "Packing failed - gripper error");
            publishRobotStatus(1, "idle", 0, 0);
            publishAvailabilityResult(1, 3, false, false, "Cart not detected");
        }
        else {
            // 로봇 작업 중
            publishRobotStatus(1, "packing", 3, 5);
            publishAvailabilityResult(1, 3, false, true, "Robot busy with another order");
        }
    }

    // Subscribers
    rclcpp::Subscription<PoseStatus>::SharedPtr pose_status_sub_;
    rclcpp::Subscription<ArmTaskStatus>::SharedPtr pick_status_sub_;
    rclcpp::Subscription<ArmTaskStatus>::SharedPtr place_status_sub_;

    // Publishers
    rclcpp::Publisher<PackingComplete>::SharedPtr packing_complete_pub_;
    rclcpp::Publisher<RobotStatus>::SharedPtr robot_status_pub_;
    rclcpp::Publisher<Availability>::SharedPtr availability_result_pub_;
    
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PackingTopicServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}