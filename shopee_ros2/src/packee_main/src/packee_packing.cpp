#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_start.hpp"
#include "shopee_interfaces/msg/packee_packing_complete.hpp"
#include "shopee_interfaces/srv/packee_vision_detect_products_in_cart.hpp"
#include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/arm_task_status.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/arm_place_product.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

using StartPacking = shopee_interfaces::srv::PackeePackingStart;
using PackingComplete = shopee_interfaces::msg::PackeePackingComplete;
using DetectProductsInCart = shopee_interfaces::srv::PackeeVisionDetectProductsInCart;
using VerifyPackingComplete = shopee_interfaces::srv::PackeeVisionVerifyPackingComplete;
using PoseStatus = shopee_interfaces::msg::ArmPoseStatus;
using robotStatus = shopee_interfaces::msg::PackeeRobotStatus;
using ArmMoveToPose = shopee_interfaces::srv::ArmMoveToPose;
using ArmPickProduct = shopee_interfaces::srv::ArmPickProduct;
using ArmPlaceProduct = shopee_interfaces::srv::ArmPlaceProduct;
using ArmTaskStatus = shopee_interfaces::msg::ArmTaskStatus;

class PackeePackingServer : public rclcpp::Node
{
public:
    PackeePackingServer() : Node("packee_packing_server")
    {
        robot_id_ = 0;
        pose_status_ = "";
        robot_status_ = "";

        start_packing_service_ = this->create_service<StartPacking>(
            "/packee/packing/start",
            std::bind(&PackeePackingServer::startPackingCallback, this, _1, _2)
        );

        pose_status_sub_ = this->create_subscription<PoseStatus>(
            "/packee/arm/pose_status",
            10,
            std::bind(&PackeePackingServer::PoseStatusCallback, this, _1)
        );

        robot_status_sub_ = this->create_subscription<robotStatus>(
            "/packee/set_robot_status", 
            10, 
            std::bind(&PackeePackingServer::StatusCallback, this, _1)
        );

        robot_status_pub_ = this->create_publisher<robotStatus>("/packee/robot_status", 10);

        packing_complete_pub_ = this->create_publisher<PackingComplete>("/packee/packing_complete", 10);

        verify_packing_client_ = this->create_client<VerifyPackingComplete>("/packee/vision/verify_packing_complete");

        move_pose_client1_ = this->create_client<ArmMoveToPose>("/packee1/arm/move_to_pose");
        move_pose_client2_ = this->create_client<ArmMoveToPose>("/packee2/arm/move_to_pose");

        RCLCPP_INFO(this->get_logger(), "PackeePackingServer started");
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

    // 포장 시작
    void startPackingCallback(
        const std::shared_ptr<StartPacking::Request> request,
        std::shared_ptr<StartPacking::Response> response)
    {
        response->success = true;
        response->message = "Packing started";

        rclcpp::Rate rate(10);
        bool result = false;
        std::string arm_side = "";
        int32_t index = 1;

        if (robot_status_ == "CHECKING_CART") {
            RCLCPP_INFO(this->get_logger(), "장바구니 확인자세 이동");

            callMovePoseLeft(request->robot_id, request->order_id, "cart_view");

            while (rclcpp::ok()) {
                if (pose_status_ == "complete") break;
                rate.sleep();
            }

            callMovePoseRight(request->robot_id, request->order_id, "cart_view");

            while (rclcpp::ok()) {
                if (pose_status_ == "complete") break;
                rate.sleep();
            }

            for (auto &product : request->products)
            {
                RCLCPP_INFO(this->get_logger(), "상품 pick & place");

                int32_t product_id = product.product_id;
                publishRobotStatus(request->robot_id, "DETECTED_PRODUCT", request->order_id, 0);

                if (index % 2 == 0) {
                    arm_side = "right";
                } else {
                    arm_side = "left";
                }

                publishRobotStatus(request->robot_id, "PACKING_PRODUCT", request->order_id, 0);
                
                result = callVerifyPacking(request->robot_id, request->order_id);
                index++;
            }

            if (result == true) {
                publishPackingComplete(request->robot_id, request->order_id, true, request->products.size(), "Sucess Packing");
            } else {
                publishPackingComplete(request->robot_id, request->order_id, false, request->products.size(), "Failed Packing");
            }

            RCLCPP_INFO(this->get_logger(), "상품 완료 포장완료");

            callMovePoseLeft(request->robot_id, request->order_id, "standby");

            while (rclcpp::ok()) {
                if (pose_status_ == "complete") break;
                rate.sleep();
            }

            callMovePoseRight(request->robot_id, request->order_id, "standby");

            while (rclcpp::ok()) {
                if (pose_status_ == "complete") break;
                rate.sleep();
            }

            publishRobotStatus(request->robot_id, "STANDBY", request->order_id, 0);
            RCLCPP_INFO(this->get_logger(), "대기자세");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "로봇 상태가 올바르지 않습니다.");
            publishPackingComplete(request->robot_id, request->order_id, false, request->products.size(), "Failed Packing");
        }
    }

    bool callMovePoseLeft(int32_t robot_id, int32_t order_id, const std::string& pose_type)
    {
        auto request = std::make_shared<ArmMoveToPose::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->pose_type = pose_type;

        while (!move_pose_client1_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "MovePoseLeft service unavailable, waiting again...");

        }

        auto future = move_pose_client1_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "MovePose success: %s, message: %s",
                        response->success ? "true" : "false",
                        response->message.c_str());
            return response->success;
        }
        return false;
    }

    bool callMovePoseRight(int32_t robot_id, int32_t order_id, const std::string& pose_type)
    {
        auto request = std::make_shared<ArmMoveToPose::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->pose_type = pose_type;

        while (!move_pose_client2_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "MovePoseRight service unavailable, waiting again...");

        }

        auto future = move_pose_client2_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "MovePose success: %s, message: %s",
                        response->success ? "true" : "false",
                        response->message.c_str());
            return response->success;
        }
        return false;
    }

    bool callVerifyPacking(int32_t robot_id, int32_t order_id)
    {
        auto request = std::make_shared<VerifyPackingComplete::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;

        if (!verify_packing_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "VerifyPacking service not available");
            return false;
        }

        auto future = verify_packing_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), 
                        "Packing complete: %s, remaining_items: %d, message: %s",
                        response->cart_empty ? "true" : "false",
                        response->remaining_items,
                        response->message.c_str());
            return response->cart_empty;
        }

        RCLCPP_ERROR(this->get_logger(), "Failed to call verify packing service");
        return false;
    }

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

private:
    int robot_id_;
    std::string pose_status_;
    std::string robot_status_;

    rclcpp::Service<StartPacking>::SharedPtr start_packing_service_;
    rclcpp::Subscription<PoseStatus>::SharedPtr pose_status_sub_;
    rclcpp::Publisher<robotStatus>::SharedPtr robot_status_pub_;
    rclcpp::Subscription<robotStatus>::SharedPtr robot_status_sub_;
    rclcpp::Publisher<PackingComplete>::SharedPtr packing_complete_pub_;
    rclcpp::Client<VerifyPackingComplete>::SharedPtr verify_packing_client_;
    rclcpp::Client<ArmMoveToPose>::SharedPtr move_pose_client1_;
    rclcpp::Client<ArmMoveToPose>::SharedPtr move_pose_client2_;

};

// ===================== main =====================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PackeePackingServer>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
