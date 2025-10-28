#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_check_availability.hpp"
#include "shopee_interfaces/srv/vision_check_cart_presence.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/packee_availability.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

using CheckAvailability = shopee_interfaces::srv::PackeePackingCheckAvailability;
using CheckCartPresence = shopee_interfaces::srv::VisionCheckCartPresence;
using ArmMoveToPose = shopee_interfaces::srv::ArmMoveToPose;
using PoseStatus = shopee_interfaces::msg::ArmPoseStatus;
using AvailabilityResult = shopee_interfaces::msg::PackeeAvailability;

class PackeePackingCheckAvailability : public rclcpp::Node
{
public:
    PackeePackingCheckAvailability() : Node("packee_packing_check_availability_server")
    {
        robot_status_ = this->declare_parameter<std::string>("robot_status", "STANBY");

        check_avail_service_ = this->create_service<CheckAvailability>(
            "/packee/packing/check_availability",
            std::bind(&PackeePackingCheckAvailability::checkAvailabilityCallback, this, _1, _2)
        );

        check_avail_pub_ = this->create_publisher<AvailabilityResult>("/packee/availability_result", 10);

        pose_status_sub_ = this->create_subscription<PoseStatus>(
            "/packee/arm/pose_status",
            10,
            std::bind(&PackeePackingCheckAvailability::PoseStatusCallback, this, _1)
        );

        RCLCPP_INFO(this->get_logger(), "packee_packing_check_availability_server started");
    }

    void PoseStatusCallback(const PoseStatus::SharedPtr msg)
    {
        robot_id_ = msg->robot_id;
        pose_status_ = msg->status;
    }

    void checkAvailabilityCallback(
        const std::shared_ptr<CheckAvailability::Request> request,
        std::shared_ptr<CheckAvailability::Response> response)
    {

        response->success = true;
        response->message = "packee check availability";

        robot_status_ = "CHECKING_CART";

        callMovePose(request->robot_id, request->order_id, "cart_view");

        if (robot_id_ == request->robot_id && pose_status_ == "complete") {
            bool cart_present = false;
            int count = 0;

            while (count < 5) {
                cart_present = callCheckCartPresence(request->robot_id);
                if (cart_present) break;
                rclcpp::sleep_for(1s);
                count++;
            }

            auto msg = AvailabilityResult();
            msg.robot_id = request->robot_id;
            msg.order_id = request->order_id;
            msg.available = robot_status_ == "CHECKING_CART" ?  true : false;
            msg.cart_detected = cart_present ? true : false;
            msg.message = cart_present ? "packee availability OK" : "packee availability NO";
            check_avail_pub_->publish(msg);
        }
    }

    bool callCheckCartPresence(int32_t robot_id)
    {
        auto request = std::make_shared<CheckCartPresence::Request>();
        request->robot_id = robot_id;

        int domain_id = robot_id == 1 ? 20 : 21;

        std::string domain = "/domain" + std::to_string(domain_id);
        cart_presence_client_ = this->create_client<CheckCartPresence>("/packee/vision/check_cart_presence");

        auto future = cart_presence_client_->async_send_request(request);
        if (future.wait_for(5s) == std::future_status::ready)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), 
                "Cart present: %s, confidence: %.2f, message: %s",
                response->cart_present ? "true" : "false",
                response->confidence,
                response->message.c_str());
            return response->cart_present;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to call cart presence service");
            return false;
        }
    }

    bool callMovePose(int32_t robot_id, int32_t order_id, const std::string& pose_type)
    {
        RCLCPP_INFO(this->get_logger(), "test1");
        auto request = std::make_shared<ArmMoveToPose::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->pose_type = pose_type;

        move_pose_client_ = this->create_client<ArmMoveToPose>("/packee" + robot_id + "/arm/move_to_pose");

        auto future = move_pose_client_->async_send_request(request);
        if (future.wait_for(5s) == std::future_status::ready)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(),
                "MovePose success: %s, message: %s",
                response->success ? "true" : "false",
                response->message.c_str());
            return response->success;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "MovePose service timeout or failed");
            return false;
        }
    }

private:
    int robot_id_;
    std::string pose_status_;
    std::string robot_status_;

    rclcpp::Service<CheckAvailability>::SharedPtr check_avail_service_;
    rclcpp::Subscription<PoseStatus>::SharedPtr pose_status_sub_;
    rclcpp::Client<CheckCartPresence>::SharedPtr cart_presence_client_;
    rclcpp::Client<ArmMoveToPose>::SharedPtr move_pose_client_;
    rclcpp::Publisher<AvailabilityResult>::SharedPtr check_avail_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PackeePackingCheckAvailability>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}