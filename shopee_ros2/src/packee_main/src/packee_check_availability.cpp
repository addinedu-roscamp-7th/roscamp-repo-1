#include <chrono>
#include <string>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_check_availability.hpp"
#include "shopee_interfaces/srv/vision_check_cart_presence.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/msg/packee_availability.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"

using namespace std::chrono_literals;
using CheckAvailability = shopee_interfaces::srv::PackeePackingCheckAvailability;
using CheckCartPresence = shopee_interfaces::srv::VisionCheckCartPresence;
using ArmMoveToPose = shopee_interfaces::srv::ArmMoveToPose;
using AvailabilityResult = shopee_interfaces::msg::PackeeAvailability;
using RobotStatus = shopee_interfaces::msg::PackeeRobotStatus;
using PoseStatus = shopee_interfaces::msg::ArmPoseStatus;

class PackeePackingCheckAvailability : public rclcpp::Node
{
public:
    PackeePackingCheckAvailability() : Node("packee_packing_check_availability_server")
    {
        robot_status_ = "";
        pose_status_ = "";
        max_retry_count_ = 10;

        check_avail_service_ = this->create_service<CheckAvailability>(
            "/packee/packing/check_availability",
            std::bind(&PackeePackingCheckAvailability::checkAvailabilityCallback, this, std::placeholders::_1, std::placeholders::_2));

        check_avail_pub_ = this->create_publisher<AvailabilityResult>("/packee/availability_result", 10);
        robot_status_pub_ = this->create_publisher<RobotStatus>("/packee/set_robot_status", 10);

        move_pose_client_ = this->create_client<ArmMoveToPose>("/packee1/arm/move_to_pose");
        cart_presence_client_ = this->create_client<CheckCartPresence>("/packee/vision/check_cart_presence");

        robot_status_sub_ = this->create_subscription<RobotStatus>(
            "/packee/robot_status", 10,
            [this](RobotStatus::SharedPtr msg) { robot_status_ = msg->state; });

        RCLCPP_INFO(this->get_logger(), "packee_packing_check_availability_server started");
    }

private:
    void checkAvailabilityCallback(
        const std::shared_ptr<CheckAvailability::Request> request,
        std::shared_ptr<CheckAvailability::Response> response)
    {
        if (robot_status_ != "STANDBY") {
            response->success = false;
            response->message = "Robot is not in STANDBY state.";
            RCLCPP_WARN(this->get_logger(), "Robot not ready (status=%s)", robot_status_.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Starting availability check (order=%d, robot=%d)", request->order_id, request->robot_id);

        response->success = true;
        response->message = "Checking packee availability...";

        publishRobotStatus(request->robot_id, "CHECKING_CART", request->order_id, 0);

        moveToPoseAsync(request->robot_id, request->order_id, "cart_view",
            [this, request](bool move_ok) {
                if (!move_ok) {
                    RCLCPP_ERROR(this->get_logger(), "MovePose(cart_view) failed");
                    publishRobotStatus(request->robot_id, "STANDBY", request->order_id, 0);
                    return;
                }

                // 재시도 로직 시작
                checkCartPresenceWithRetry(request->robot_id, request->order_id, 0);
            });
    }

    void checkCartPresenceWithRetry(int32_t robot_id, int32_t order_id, int retry_count)
    {
        RCLCPP_INFO(this->get_logger(), "Cart detection attempt %d/%d", retry_count + 1, max_retry_count_);

        checkCartPresenceAsync(robot_id, order_id,
            [this, robot_id, order_id, retry_count](bool cart_present) {
                if (cart_present) {
                    // 카트 감지 성공 - 대기 상태로 복귀
                    RCLCPP_INFO(this->get_logger(), "Cart detected on attempt %d", retry_count + 1);
                } else {
                    // 카트 감지 실패
                    if (retry_count + 1 >= max_retry_count_) {
                        // 최대 재시도 횟수 도달 - 대기 상태로 복귀
                        RCLCPP_WARN(this->get_logger(), "Cart not detected after %d attempts", max_retry_count_);
                        returnToStandby(robot_id, order_id, false);
                    } else {
                        // 재시도
                        RCLCPP_WARN(this->get_logger(), "Cart not detected, retrying...");
                        checkCartPresenceWithRetry(robot_id, order_id, retry_count + 1);
                    }
                }
            });
    }

    void returnToStandby(int32_t robot_id, int32_t order_id, bool cart_detected)
    {
        moveToPoseAsync(robot_id, order_id, "standby",
            [this, robot_id, order_id, cart_detected](bool move_back_ok) {
                if (!move_back_ok) {
                    RCLCPP_ERROR(this->get_logger(), "MovePose(standby) failed");
                }

                publishRobotStatus(robot_id, "STANDBY", order_id, 0);

                auto msg = AvailabilityResult();
                msg.robot_id = robot_id;
                msg.order_id = order_id;
                msg.available = cart_detected;
                msg.cart_detected = cart_detected;
                msg.message = cart_detected
                                ? "Packee availability OK (cart detected)"
                                : "No cart detected after maximum retries";
                check_avail_pub_->publish(msg);

                RCLCPP_INFO(this->get_logger(),
                    "Availability check done (cart_detected=%s)",
                    cart_detected ? "true" : "false");
            });
    }

    void moveToPoseAsync(int32_t robot_id, int32_t order_id, const std::string &pose_type,
                         std::function<void(bool)> done_cb)
    {
        if (!move_pose_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "MovePose service not available.");
            done_cb(false);
            return;
        }

        auto req = std::make_shared<ArmMoveToPose::Request>();
        req->robot_id = robot_id;
        req->order_id = order_id;
        req->pose_type = pose_type;

        move_pose_client_->async_send_request(req,
            [this, pose_type, done_cb](rclcpp::Client<ArmMoveToPose>::SharedFuture future) {
                auto res = future.get();
                RCLCPP_INFO(this->get_logger(),
                    "MovePose(%s): %s (%s)",
                    pose_type.c_str(),
                    res->success ? "true" : "false",
                    res->message.c_str());
                done_cb(res->success);
            });
    }

    void checkCartPresenceAsync(int32_t robot_id, int32_t order_id,
                                std::function<void(bool)> done_cb)
    {
        if (!cart_presence_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "CartPresence service not available.");
            done_cb(false);
            return;
        }

        auto req = std::make_shared<CheckCartPresence::Request>();
        req->robot_id = robot_id;

        cart_presence_client_->async_send_request(req,
            [this, robot_id, order_id, done_cb](rclcpp::Client<CheckCartPresence>::SharedFuture future) {
                auto res = future.get();
                RCLCPP_INFO(this->get_logger(),
                    "Cart presence: %s (conf=%.2f, msg=%s)",
                    res->cart_present ? "true" : "false",
                    res->confidence,
                    res->message.c_str());
                done_cb(res->cart_present);
            });
    }

    void publishRobotStatus(int id, const std::string &state, int order_id, int items_in_cart)
    {
        RobotStatus msg;
        msg.robot_id = id;
        msg.state = state;
        msg.current_order_id = order_id;
        msg.items_in_cart = items_in_cart;
        robot_status_pub_->publish(msg);
    }

    std::string robot_status_;
    std::string pose_status_;
    int max_retry_count_;

    rclcpp::Service<CheckAvailability>::SharedPtr check_avail_service_;
    rclcpp::Publisher<AvailabilityResult>::SharedPtr check_avail_pub_;
    rclcpp::Publisher<RobotStatus>::SharedPtr robot_status_pub_;
    rclcpp::Subscription<PoseStatus>::SharedPtr pose_status_sub_;
    rclcpp::Subscription<RobotStatus>::SharedPtr robot_status_sub_;
    rclcpp::Client<ArmMoveToPose>::SharedPtr move_pose_client_;
    rclcpp::Client<CheckCartPresence>::SharedPtr cart_presence_client_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PackeePackingCheckAvailability>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}