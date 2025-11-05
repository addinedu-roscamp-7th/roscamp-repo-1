#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_check_availability.hpp"
#include "shopee_interfaces/srv/vision_check_cart_presence.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/arm_task_status.hpp"
#include "shopee_interfaces/msg/packee_availability.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

using CheckAvailability = shopee_interfaces::srv::PackeePackingCheckAvailability;
using CheckCartPresence = shopee_interfaces::srv::VisionCheckCartPresence;
using ArmMoveToPose = shopee_interfaces::srv::ArmMoveToPose;
using PoseStatus = shopee_interfaces::msg::ArmPoseStatus;
using AvailabilityResult = shopee_interfaces::msg::PackeeAvailability;
using robotStatus = shopee_interfaces::msg::PackeeRobotStatus;

class PackeePackingCheckAvailability : public rclcpp::Node
{
public:
    PackeePackingCheckAvailability() : Node("packee_packing_check_availability_server")
    {
        robot_id_ = 0;
        pose_status_ = "";
        robot_status_ = "";

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

        robot_status_sub_ = this->create_subscription<robotStatus>("/packee/robot_status", 10, std::bind(&PackeePackingCheckAvailability::StatusCallback, this, _1));

        robot_status_pub_ = this->create_publisher<robotStatus>("/packee/set_robot_status", 10);

        cart_presence_client_ = this->create_client<CheckCartPresence>("/packee/vision/check_cart_presence");
        move_pose_client_ = this->create_client<ArmMoveToPose>("/packee1/arm/move_to_pose");

        RCLCPP_INFO(this->get_logger(), "packee_packing_check_availability_server started");
    }

    void StatusCallback(const robotStatus::SharedPtr msg)
    {
        robot_status_ = msg->state;
    }

    void PoseStatusCallback(const PoseStatus::SharedPtr msg)
    {
        robot_id_ = msg->robot_id;
        pose_status_ = msg->status;
    }

    void publishRobotStatus(int id, const std::string &state, int order_id, int items_in_cart)
    {
        auto msg = robotStatus();
        msg.robot_id = id;
        msg.state = state;
        msg.current_order_id = order_id;
        msg.items_in_cart = items_in_cart;
        robot_status_pub_->publish(msg);
    }

    void checkAvailabilityCallback(
        const std::shared_ptr<CheckAvailability::Request> request,
        std::shared_ptr<CheckAvailability::Response> response)
    {
        int32_t robot_id = request->robot_id;
        int32_t order_id = request->order_id;

        if (robot_status_ == "STANDBY") {
            response->success = true;
            response->message = "Checking packee availability...";

            publishRobotStatus(request->robot_id, "CHECKING_CART", request->order_id, 0);

            // 1. 카트 뷰 위치로 이동
            if (!callMovePose(robot_id, order_id, "cart_view")) {
                RCLCPP_ERROR(this->get_logger(), "MovePose(cart_view) failed");
                response->success = false;
                response->message = "MovePose(cart_view) failed";
                return;
            }

            // 2. Pose 완료 대기
            rclcpp::Rate rate(10);
            while (rclcpp::ok()) {
                if (pose_status_ == "complete") break;
                rate.sleep();
            }

            // 3. 카트 감지 반복 시도
            bool cart_present = false;
            int retry = 0;
            while (!cart_present && retry < 5 && rclcpp::ok()) {
                cart_present = callCheckCartPresence(robot_id);
                RCLCPP_INFO(this->get_logger(), "Cart check retry %d/5", retry + 1);
                rclcpp::sleep_for(1s);
                retry++;
            }

            // 4. standby 위치로 복귀
            if (!callMovePose(robot_id, order_id, "standby")) {
                RCLCPP_ERROR(this->get_logger(), "MovePose(standby) failed");
                response->success = false;
                response->message = "MovePose(cart_view) failed";
                return;
            }

            while (rclcpp::ok()) {
                if (pose_status_ == "complete") break;
                rate.sleep();
            }

            // 5. 상태 변경
            publishRobotStatus(request->robot_id, "STANDBY", request->order_id, 0);

            // 6. 결과 Publish
            auto msg = AvailabilityResult();
            msg.robot_id = robot_id;
            msg.order_id = order_id;
            msg.available = cart_present;
            msg.cart_detected = cart_present;
            msg.message = cart_present ? "Packee availability OK (cart detected)" : "No cart detected";
            check_avail_pub_->publish(msg);

            RCLCPP_INFO(this->get_logger(),
                "Availability Result → detected=%s, message=%s",
                cart_present ? "true" : "false",
                msg.message.c_str());

        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "로봇 상태가 올바르지 않습니다.");
            response->success = false;
            response->message = "check availability failed";
        }

        
    }

    bool callCheckCartPresence(int32_t robot_id)
    {
        auto request = std::make_shared<CheckCartPresence::Request>();
        request->robot_id = robot_id;

        while (!cart_presence_client_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "CheckCartPresence service unavailable, waiting again...");

        }

        auto future = cart_presence_client_->async_send_request(request);
        if (future.wait_for(5s) == std::future_status::ready)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(),
                "Cart present: %s (conf=%.2f, msg=%s)",
                response->cart_present ? "true" : "false",
                response->confidence,
                response->message.c_str());
            return response->cart_present;
        }

        RCLCPP_ERROR(this->get_logger(), "Failed to call cart presence service");
        return false;
    }

    bool callMovePose(int32_t robot_id, int32_t order_id, const std::string &pose_type)
    {
        auto request = std::make_shared<ArmMoveToPose::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->pose_type = pose_type;

        while (!move_pose_client_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "MovePose service unavailable, waiting again...");

        }

        auto future = move_pose_client_->async_send_request(request);
        if (future.wait_for(5s) == std::future_status::ready)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(),
                "MovePose(%s): %s (%s)",
                pose_type.c_str(),
                response->success ? "true" : "false",
                response->message.c_str());

            return response->success;
        }

        RCLCPP_ERROR(this->get_logger(), "MovePose(%s) call failed", pose_type.c_str());
        return false;
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
    rclcpp::Publisher<robotStatus>::SharedPtr robot_status_pub_;
    rclcpp::Subscription<robotStatus>::SharedPtr robot_status_sub_;
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