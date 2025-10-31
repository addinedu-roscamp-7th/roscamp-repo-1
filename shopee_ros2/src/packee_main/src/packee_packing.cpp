#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_start.hpp"
#include "shopee_interfaces/msg/packee_packing_complete.hpp"
#include "shopee_interfaces/srv/packee_vision_detect_products_in_cart.hpp"
#include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/arm_place_product.hpp"
#include "packee_state_manager.hpp" 

using namespace std::chrono_literals;
using namespace std::placeholders;

using StartPacking = shopee_interfaces::srv::PackeePackingStart;
using CheckCartPresence = shopee_interfaces::srv::VisionCheckCartPresence;
using DetectProductsInCart = shopee_interfaces::srv::PackeeVisionDetectProductsInCart;
using VerifyPackingComplete = shopee_interfaces::srv::PackeeVisionVerifyPackingComplete;
using PoseStatus = shopee_interfaces::msg::ArmPoseStatus;
using ArmMoveToPose = shopee_interfaces::srv::ArmMoveToPose;
using ArmPickProduct = shopee_interfaces::srv::ArmPickProduct;
using ArmPlaceProduct = shopee_interfaces::srv::ArmPlaceProduct;
using ArmTaskStatus = shopee_interfaces::msg::ArmTaskStatus;

class PackeePackingServer : public rclcpp::Node
{
public:
    PackeePackingServer() : Node("packee_packing_server")
    {
        state_manager_ = std::make_shared<PackeeStateManager>(this->shared_from_this());

        start_packing_service_ = this->create_service<StartPacking>(
            "/packee/packing/start",
            std::bind(&PackeePackingServer::startPackingCallback, this, _1, _2)
        );

        pick_status_sub_ = this->create_subscription<ArmTaskStatus>(
            "/packee/arm/pick_status",
            10,
            std::bind(&PackeePackingServer::PickStatusCallback, this, _1)
        );

        place_status_sub_ = this->create_subscription<ArmTaskStatus>(
            "/packee/arm/place_status",
            10,
            std::bind(&PackeePackingServer::PlaceStatusCallback, this, _1)
        );

        packing_complete_pub_ = this->create_publisher<PackingComplete>("/packee/packing_complete", 10);

        detect_products_client_ = this->create_client<DetectProductsInCart>("/packee/vision/detect_products_in_cart");
        verify_packing_client_ = this->create_client<VerifyPackingComplete>("/packee/vision/verify_packing_complete");

        move_pose_client_ = this->create_client<ArmMoveToPose>("/packee/arm/move_to_pose");
        pick_product_client_ = this->create_client<ArmPickProduct>("/packee/arm/pick_product");
        place_product_client_ = this->create_client<ArmPlaceProduct>("/packee/arm/place_product");

        RCLCPP_INFO(this->get_logger(), "PackeePackingServer started");
    }

    void PoseStatusCallback(const PoseStatus::SharedPtr msg)
    {
        robot_id_ = msg->robot_id;
        pose_status_ = msg->status;
    }

    void PickStatusCallback(const ArmTaskStatus::SharedPtr msg)
    {
        pick_arm_side = msg.arm_side;
        pick_status_ = msg.status;
    }

    void PlaceStatusCallback(const ArmTaskStatus::SharedPtr msg)
    {
        place_arm_side_ = msg.arm_side;
        place_status_ = msg.status;
    }

    // 포장 시작
    void startPackingCallback(
        const std::shared_ptr<StartPacking::Request> request,
        std::shared_ptr<StartPacking::Response> response)
    {
        response->success = true;
        response->message = "Packing started";

        bool result = false;
        if (state_manager_->getStatus() == "CHECKING_CART") {
            state_manager_->setStatus(request->robot_id, "DETECTED_PRODUCT", request->order_id, 0);

            for (auto &product_id : request->expected_product_ids)
            {
                callMovePose(request->robot_id, request->order_id, "cart_view");

                while (pose_status_ != "complete" && rclcpp::ok())
                    rclcpp::sleep_for(100ms);

                std::vector<float> pose = callDetectProducts(request->robot_id, request->order_id, product_id);

                state_manager_->setStatus(request->robot_id, "PACKING_PRODUCT", request->order_id, 0);
                callPickProduct(request->robot_id, request->order_id, product_id, "left", pose);

                while (pick_status_ != "completed" && rclcpp::ok())
                    rclcpp::sleep_for(100ms);

                callPlaceProduct(request->robot_id, request->order_id, product_id, "left", pose);

                while (place_status_ != "completed" && rclcpp::ok())
                    rclcpp::sleep_for(100ms);

                result = callVerifyPacking(request->robot_id, request->order_id);
            }


            if (result == true) {
                publishPackingComplete(request->robot_id, request->order_id, true, request->expected_product_ids, "Sucess Packing");
            } else {
                publishPackingComplete(request->robot_id, request->order_id, true, request->expected_product_ids, "Failed Packing");
            }

            callMovePose(request->robot_id, request->order_id, "standby");

            state_manager_->setStatus("STANDBY");
        }
    }

    std::vector<float> callDetectProducts(int32_t robot_id, int32_t order_id, int32_t product_id)
    {
        auto request = std::make_shared<DetectProductsInCart::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->expected_product_id = product_id;

        if (!detect_products_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "Detect products service not available");
            return false;
        }

        auto future = detect_products_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), 
                "Detection success: %s, message: %s",
                response->success ? "true" : "false",
                response->message.c_str());
            return response->pose;
        }

        RCLCPP_ERROR(this->get_logger(), "Failed to call detect products service");
        return {0, 0, 0, 0, 0, 0};
    }

    bool callMovePose(int32_t robot_id, int32_t order_id, const std::string& pose_type)
    {
        auto request = std::make_shared<ArmMoveToPose::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->pose_type = pose_type;

        if (!move_pose_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "MovePose service not available");
            return false;
        }

        auto future = move_pose_client_->async_send_request(request);
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

    bool callPickProduct(int32_t robot_id, int32_t order_id, int32_t product_id,
                         const std::string& arm_side, const std::vector<float>& target_position)
    {
        auto request = std::make_shared<ArmPickProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;

        if(target_position.size() == 6) {
            request->pose.x = target_position[0];
            request->pose.y = target_position[1];
            request->pose.z = target_position[2];
            request->pose.rx = target_position[3];
            request->pose.ry = target_position[4];
            request->pose.rz = target_position[5];
        }

        if (!pick_product_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "PickProduct service not available");
            return false;
        }

        auto future = pick_product_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "PickProduct success: %s, message: %s",
                        response->success ? "true" : "false",
                        response->message.c_str());
            return response->success;
        }
        return false;
    }

    bool callPlaceProduct(int32_t robot_id, int32_t order_id, int32_t product_id,
                          const std::string& arm_side, const std::vector<float>& box_position)
    {
        auto request = std::make_shared<ArmPlaceProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;

        if (!place_product_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "PlaceProduct service not available");
            return false;
        }

        auto future = place_product_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "PlaceProduct success: %s, message: %s",
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
    std::string arm_side_;
    std::string pick_status_;
    std::string place_status_;
    std::shared_ptr<RobotStateManager> state_manager_;

    rclcpp::Service<StartPacking>::SharedPtr start_packing_service_;

    rclcpp::Subscription<ArmTaskStatus>::SharedPtr pick_status_sub_;
    rclcpp::Subscription<ArmTaskStatus>::SharedPtr place_status_sub_;

    rclcpp::Client<DetectProductsInCart>::SharedPtr detect_products_client_;
    rclcpp::Client<VerifyPackingComplete>::SharedPtr verify_packing_client_;

    rclcpp::Client<ArmMoveToPose>::SharedPtr move_pose_client_;
    rclcpp::Client<ArmPickProduct>::SharedPtr pick_product_client_;
    rclcpp::Client<ArmPlaceProduct>::SharedPtr place_product_client_;
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
