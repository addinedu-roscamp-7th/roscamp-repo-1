#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_check_availability.hpp"
#include "shopee_interfaces/srv/vision_check_cart_presence.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/packee_availability.hpp"
#include "packee_state_manager.hpp" 

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
        state_manager_ = std::make_shared<PackeeStateManager>(this->shared_from_this());

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

        cart_presence_client_ = this->create_client<CheckCartPresence>("/packee/vision/check_cart_presence");
        move_pose_client_ = this->create_client<ArmMoveToPose>("/packee/arm/move_to_pose");

        RCLCPP_INFO(this->get_logger(), "packee_packing_check_availability_server started");
    }

    void PoseStatusCallback(const PoseStatus::SharedPtr msg)
    {
        if (msg->robot_id == robot_id_)
            pose_status_ = msg->status;
    }

    void checkAvailabilityCallback(
        const std::shared_ptr<CheckAvailability::Request> request,
        std::shared_ptr<CheckAvailability::Response> response)
    {
        int32_t robot_id = request->robot_id;
        int32_t order_id = request->order_id;

        response->success = true;
        response->message = "Checking packee availability...";

        state_manager_->setStatus(robot_id, "CHECKING_CART", order_id, 0);

        // 1. 카트 뷰 위치로 이동
        if (!callMovePose(robot_id, order_id, "cart_view")) {
            RCLCPP_ERROR(this->get_logger(), "MovePose(cart_view) failed");
            response->success = false;
            response->message = "MovePose(cart_view) failed";
            state_manager_->setStatus(robot_id, "ERROR", order_id, 0);
            return;
        }

        // 2. Pose 완료 대기
        while (pose_status_ != "complete" && rclcpp::ok())
            rclcpp::sleep_for(100ms);

        // 3. 카트 감지 반복 시도
        bool cart_present = false;
        int retry = 0;
        while (retry < 5 && rclcpp::ok()) {
            cart_present = callCheckCartPresence(robot_id);
            if (cart_present) break;
            rclcpp::sleep_for(1s);
            retry++;
        }

        // 4. standby 위치로 복귀
        if (!callMovePose(robot_id, order_id, "standby")) {
            RCLCPP_WARN(this->get_logger(), "MovePose(standby) failed");
        } else {
            while (pose_status_ != "complete" && rclcpp::ok())
                rclcpp::sleep_for(100ms);
        }

        // 5. 상태 변경
        state_manager_->setStatus(robot_id, "STANDBY", order_id, 0);

        // 6. 결과 Publish
        auto msg = AvailabilityResult();
        msg.robot_id = robot_id;
        msg.order_id = order_id;
        msg.available = true;
        msg.cart_detected = cart_present;
        msg.message = cart_present ? "Packee availability OK (cart detected)" : "No cart detected";
        check_avail_pub_->publish(msg);

        RCLCPP_INFO(this->get_logger(),
            "Availability Result → detected=%s, message=%s",
            cart_present ? "true" : "false",
            msg.message.c_str());
    }

    bool callCheckCartPresence(int32_t robot_id)
    {
        if (!cart_presence_client_->wait_for_service(2s)) {
            RCLCPP_ERROR(this->get_logger(), "CheckCartPresence service unavailable");
            return false;
        }

        auto request = std::make_shared<CheckCartPresence::Request>();
        request->robot_id = robot_id;

        auto future = cart_presence_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future)
            == rclcpp::FutureReturnCode::SUCCESS)
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
        if (!move_pose_client_->wait_for_service(2s)) {
            RCLCPP_ERROR(this->get_logger(), "MovePose service unavailable");
            return false;
        }

        auto request = std::make_shared<ArmMoveToPose::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->pose_type = pose_type;

        auto future = move_pose_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future)
            == rclcpp::FutureReturnCode::SUCCESS)
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
    int robot_id_{0};
    std::string pose_status_;

    std::shared_ptr<PackeeStateManager> state_manager_;
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