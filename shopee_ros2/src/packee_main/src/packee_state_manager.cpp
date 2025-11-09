#include "rclcpp/rclcpp.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"

using robotStatus = shopee_interfaces::msg::PackeeRobotStatus;
using namespace std::placeholders;

class PackeeStateManager : public rclcpp::Node
{
public:
    PackeeStateManager() : Node("Robot_Manager_Server")
    {
        robot_id_ = 0;
        current_status_ = "STANDBY";
        current_order_id_ = 0;
        items_in_cart_ = 0;

        state_sub_ = this->create_subscription<robotStatus>(
            "/packee/set_robot_status", 
            10,
            std::bind(&PackeeStateManager::statusCallback, this, _1)
        );

        state_pub_ = this->create_publisher<robotStatus>("/packee/robot_status", 10);

        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&PackeeStateManager::sendStatus, this)
        );

        RCLCPP_INFO(this->get_logger(), "Robot Status Manager started");
    }

    void statusCallback(const robotStatus::SharedPtr msg)
    {
        robot_id_ = msg->robot_id;
        current_status_ = msg->state;
        current_order_id_ = msg->current_order_id;
        items_in_cart_ = msg->items_in_cart;
        
        RCLCPP_INFO(this->get_logger(), "[RECEIVED] Robot %d status: %s", 
                    robot_id_, current_status_.c_str());
    }

    void sendStatus()
    {
        auto msg = robotStatus();
        msg.robot_id = robot_id_;
        msg.state = current_status_;
        msg.current_order_id = current_order_id_;
        msg.items_in_cart = items_in_cart_;
        state_pub_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "[SENT] Robot status → %s", current_status_.c_str());
    }

private:
    int32_t robot_id_;
    std::string current_status_;
    int32_t current_order_id_;
    int32_t items_in_cart_;

    rclcpp::Subscription<robotStatus>::SharedPtr state_sub_;
    rclcpp::Publisher<robotStatus>::SharedPtr state_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PackeeStateManager>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}