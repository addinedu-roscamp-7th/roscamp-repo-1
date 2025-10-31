#include "rclcpp/rclcpp.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"

using robotStatus = shopee_interfaces::msg::PackeeRobotStatus;

class PackeeStateManager
{
public:
    PackeeStateManager(rclcpp::Node::SharedPtr node) : node_(node)
    {
        state_pub_ = node_->create_publisher<robotStatus>("/packee/robot_status", 10);
        setStatus(2, "STANDBY", 0, 0);
    }

    void setStatus(int32_t robot_id, const std::string &status, int32_t current_order_id, int32_t items_in_cart)
    {
        current_status_ = status;
        auto msg = robotStatus();
        msg.robot_id = robot_id;
        msg.status = status;
        msg.current_order_id = current_order_id;
        msg.items_in_cart = items_in_cart;
        state_pub_->publish(msg);
        RCLCPP_INFO(node_->get_logger(), "[STATE] Robot status → %s", status.c_str());
    }

    std::string getStatus() const
    {
        return current_status_;
    }

    bool is(const std::string &status) const
    {
        return current_status_ == status;
    }

private:
    std::string current_status_;
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<robotStatus>::SharedPtr state_pub_;
};
