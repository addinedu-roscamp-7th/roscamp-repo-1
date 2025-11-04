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
        pick_arm_side_ = "";
        pick_status_ = "";
        place_arm_side_ = "";
        place_status_ = "";
        robot_status_ = "";

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

        pose_status_sub_ = this->create_subscription<PoseStatus>(
            "/packee/arm/pose_status",
            10,
            std::bind(&PackeePackingServer::PoseStatusCallback, this, _1)
        );

        robot_status_pub_ = this->create_publisher<robotStatus>("/packee/robot_status", 10);

        packing_complete_pub_ = this->create_publisher<PackingComplete>("/packee/packing_complete", 10);

        detect_products_client_ = this->create_client<DetectProductsInCart>("/packee/vision/detect_products_in_cart");
        verify_packing_client_ = this->create_client<VerifyPackingComplete>("/packee/vision/verify_packing_complete");

        move_pose_client1_ = this->create_client<ArmMoveToPose>("/packee1/arm/move_to_pose");
        move_pose_client2_ = this->create_client<ArmMoveToPose>("/packee2/arm/move_to_pose");
        pick_product_client1_ = this->create_client<ArmPickProduct>("/packee1/arm/pick_product");
        pick_product_client2_ = this->create_client<ArmPickProduct>("/packee2/arm/pick_product");
        place_product_client1_ = this->create_client<ArmPlaceProduct>("/packee1/arm/place_product");
        place_product_client2_ = this->create_client<ArmPlaceProduct>("/packee2/arm/place_product");

        RCLCPP_INFO(this->get_logger(), "PackeePackingServer started");
    }

    struct DetectionResult
    {
        std::string arm_side;
        std::vector<float> pose;
    };

    void PoseStatusCallback(const PoseStatus::SharedPtr msg)
    {
        robot_id_ = msg->robot_id;
        pose_status_ = msg->status;
    }

    void PickStatusCallback(const ArmTaskStatus::SharedPtr msg)
    {
        pick_arm_side_ = msg->arm_side;
        pick_status_ = msg->status;
    }

    void PlaceStatusCallback(const ArmTaskStatus::SharedPtr msg)
    {
        place_arm_side_ = msg->arm_side;
        place_status_ = msg->status;
    }

    void publishRobotStatus(int id, const std::string &state, int order_id, int items_in_cart)
    {
        robot_status_ = state;
        auto msg = robotStatus();
        msg.robot_id = id;
        msg.state = state;
        msg.current_order_id = order_id;
        msg.items_in_cart = items_in_cart;
        robot_status_pub_->publish(msg);
    }

    float pose_diff_norm(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty()) return 1e9f;
        float sum = 0.f;
        for (size_t i = 0; i < a.size(); ++i)
            sum += std::pow(a[i] - b[i], 2);
        return std::sqrt(sum);
    }

    // 포장 시작
    void startPackingCallback(
        const std::shared_ptr<StartPacking::Request> request,
        std::shared_ptr<StartPacking::Response> response)
    {
        response->success = true;
        response->message = "Packing started";

        bool result = false;
        bool complete_flag = false;
        std::string arm_side;
        std::vector<float> current_pose;
        std::vector<float> prev_pose;
        rclcpp::Rate rate(10);

        if (robot_status_ == "CHECKING_CART") {
            publishRobotStatus(request->robot_id, "DETECTED_PRODUCT", request->order_id, 0);

            for (auto &product : request->products)
            {
                int32_t product_id = product.product_id;
                
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

                publishRobotStatus(request->robot_id, "PACKING_PRODUCT", request->order_id, 0);

                while (true)
                {
                    DetectionResult results = callDetectProducts(request->robot_id, request->order_id, product_id);

                    arm_side = results.arm_side;
                    current_pose = results.pose;

                    if (!prev_pose.empty()) {
                        float diff = pose_diff_norm(current_pose, prev_pose);

                        if (diff < 10.0f) {
                            complete_flag = true;
                            current_pose[2] -= 40;
                        }
                    }

                    if (arm_side == "left") {
                        callPickProductLeft(request->robot_id, request->order_id, product_id, "left", current_pose);

                        while (rclcpp::ok()) {
                            if (pick_status_ == "complete") break;
                            rate.sleep();
                        }
                    }
                    else if (arm_side == "right")
                    {
                        callPickProductRight(request->robot_id, request->order_id, product_id, "right", current_pose);

                        while (rclcpp::ok()) {
                            if (pick_status_ == "complete") break;
                            rate.sleep();
                        }
                    }
                    else
                    {
                        RCLCPP_ERROR(this->get_logger(), "유효하지 않은 정보입니다.");
                    }

                    prev_pose = current_pose;

                    if (complete_flag) break;
                }
                
                // place 코드

                result = callVerifyPacking(request->robot_id, request->order_id);
            }

            if (result == true) {
                publishPackingComplete(request->robot_id, request->order_id, true, request->products.size(), "Sucess Packing");
            } else {
                publishPackingComplete(request->robot_id, request->order_id, false, request->products.size(), "Failed Packing");
            }

            if (arm_side == "left")
            {
                callMovePoseLeft(request->robot_id, request->order_id, "standby");

                while (rclcpp::ok()) {
                    if (pose_status_ == "complete") break;
                    rate.sleep();
                }
            }
            else if (arm_side == "right")
            {
                callMovePoseRight(request->robot_id, request->order_id, "standby");

                while (rclcpp::ok()) {
                    if (pose_status_ == "complete") break;
                    rate.sleep();
                }
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "유효하지 않은 정보입니다.");
            }

            publishRobotStatus(request->robot_id, "STANDBY", request->order_id, 0);
        }
    }

    DetectionResult callDetectProducts(int32_t robot_id, int32_t order_id, int32_t product_id)
    {
        DetectionResult result;
        auto request = std::make_shared<DetectProductsInCart::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->expected_product_id = product_id;

        while (!detect_products_client_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "MovePoseLeft service unavailable, waiting again...");

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

            result.arm_side = response->arm_side;
            auto &p = response->products[0]; // 첫 번째 감지된 상품
            result.pose = {
                p.pose.x,
                p.pose.y,
                p.pose.z,
                p.pose.rx,
                p.pose.ry,
                p.pose.rz
            };

            return result;
        }

        RCLCPP_ERROR(this->get_logger(), "Failed to call detect products service");
        DetectionResult response;
        response.pose = {0, 0, 0, 0, 0, 0};
        return response;
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

    bool callPickProductLeft(int32_t robot_id, int32_t order_id, int32_t product_id,
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

        while (!pick_product_client1_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "PickProductLeft service unavailable, waiting again...");

        }

        auto future = pick_product_client1_->async_send_request(request);
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

    bool callPickProductRight(int32_t robot_id, int32_t order_id, int32_t product_id,
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

        while (!pick_product_client2_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "PickProductRight service unavailable, waiting again...");

        }

        auto future = pick_product_client2_->async_send_request(request);
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

    bool callPlaceProductLeft(int32_t robot_id, int32_t order_id, int32_t product_id,
                          const std::string& arm_side, const std::vector<float>& box_position)
    {
        auto request = std::make_shared<ArmPlaceProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;

        while (!place_product_client1_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "PlaceProductLeft service unavailable, waiting again...");

        }

        auto future = place_product_client1_->async_send_request(request);
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

    bool callPlaceProductRight(int32_t robot_id, int32_t order_id, int32_t product_id,
                          const std::string& arm_side, const std::vector<float>& box_position)
    {
        auto request = std::make_shared<ArmPlaceProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;

        while (!place_product_client2_->wait_for_service(1s)) 
        {
            if (!rclcpp::ok())
            {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            }
            RCLCPP_INFO(this->get_logger(), "PlaceProductRight service unavailable, waiting again...");

        }

        auto future = place_product_client2_->async_send_request(request);
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
    std::string pick_arm_side_;
    std::string pick_status_;
    std::string place_arm_side_;
    std::string place_status_;
    std::string robot_status_;

    rclcpp::Service<StartPacking>::SharedPtr start_packing_service_;

    rclcpp::Subscription<ArmTaskStatus>::SharedPtr pick_status_sub_;
    rclcpp::Subscription<ArmTaskStatus>::SharedPtr place_status_sub_;
    rclcpp::Subscription<PoseStatus>::SharedPtr pose_status_sub_;

    rclcpp::Publisher<robotStatus>::SharedPtr robot_status_pub_;
    rclcpp::Publisher<PackingComplete>::SharedPtr packing_complete_pub_;

    rclcpp::Client<DetectProductsInCart>::SharedPtr detect_products_client_;
    rclcpp::Client<VerifyPackingComplete>::SharedPtr verify_packing_client_;

    rclcpp::Client<ArmMoveToPose>::SharedPtr move_pose_client1_;
    rclcpp::Client<ArmMoveToPose>::SharedPtr move_pose_client2_;
    rclcpp::Client<ArmPickProduct>::SharedPtr pick_product_client1_;
    rclcpp::Client<ArmPickProduct>::SharedPtr pick_product_client2_;
    rclcpp::Client<ArmPlaceProduct>::SharedPtr place_product_client1_;
    rclcpp::Client<ArmPlaceProduct>::SharedPtr place_product_client2_;


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
