#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_check_availability.hpp"
#include "shopee_interfaces/srv/packee_packing_start.hpp"
#include "shopee_interfaces/srv/vision_check_cart_presence.hpp"
#include "shopee_interfaces/srv/packee_vision_detect_products_in_cart.hpp"
#include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/arm_place_product.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

using CheckAvailability = shopee_interfaces::srv::PackeePackingCheckAvailability;
using StartPacking = shopee_interfaces::srv::PackeePackingStart;
using CheckCartPresence = shopee_interfaces::srv::VisionCheckCartPresence;
using DetectProductsInCart = shopee_interfaces::srv::PackeeVisionDetectProductsInCart;
using VerifyPackingComplete = shopee_interfaces::srv::PackeeVisionVerifyPackingComplete;
using ArmMoveToPose = shopee_interfaces::srv::ArmMoveToPose;
using ArmPickProduct = shopee_interfaces::srv::ArmPickProduct;
using ArmPlaceProduct = shopee_interfaces::srv::ArmPlaceProduct;

class PackingServiceServer : public rclcpp::Node
{
public:
    PackingServiceServer() : Node("packing_service_server")
    {
        check_avail_service_ = this->create_service<CheckAvailability>(
            "/packee/packing/check_availability",
            std::bind(&PackingServiceServer::checkAvailabilityCallback, this, _1, _2)
        );

        start_packing_service_ = this->create_service<StartPacking>(
            "/packee/packing/start",
            std::bind(&PackingServiceServer::startPackingCallback, this, _1, _2)
        );

        std::string robot_status = this->declare_parameter<std::string>("robot_status", "CHECKING_CART");

        // 클라이언트
        cart_presence_client_ = this->create_client<CheckCartPresence>("/packee/vision/check_cart_presence");
        detect_products_client_ = this->create_client<DetectProductsInCart>("/packee/vision/detect_products_in_cart");
        verify_packing_client_ = this->create_client<VerifyPackingComplete>("/packee/vision/verify_packing_complete");

        move_pose_client_ = this->create_client<ArmMoveToPose>("/packee/arm/move_to_pose");
        pick_product_client_ = this->create_client<ArmPickProduct>("/packee/arm/pick_product");
        place_product_client_ = this->create_client<ArmPlaceProduct>("/packee/arm/place_product");

        RCLCPP_INFO(this->get_logger(), "PackingServiceServer started");
    }

    // ================= 서비스 콜백 =================
    void checkAvailabilityCallback(
        const std::shared_ptr<CheckAvailability::Request> request,
        std::shared_ptr<CheckAvailability::Response> response)
    {
        // robot_status = "";
        response->success = true;
        response->message = "Availability check OK";
    }

    void startPackingCallback(
        const std::shared_ptr<StartPacking::Request> request,
        std::shared_ptr<StartPacking::Response> response)
    {
        response->success = true;
        response->message = "Packing started";
        
        // TODO
        // if (robot_status == "CHECKING_CART") {
        //     robot_status = "Detecting_Products";
            
        //     callDetectProducts(request->robot_id, request->order_id, request->expected_product_ids);
        // }
    }

    // ================= 클라이언트 호출 함수 =================
    bool callCheckCartPresence(int32_t robot_id)
    {
        auto request = std::make_shared<CheckCartPresence::Request>();
        request->robot_id = robot_id;

        if (!cart_presence_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "Cart presence service not available");
            return false;
        }

        auto future = cart_presence_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), 
                "Cart present: %s, confidence: %.2f, message: %s",
                response->cart_present ? "true" : "false",
                response->confidence,
                response->message.c_str());
            return response->cart_present;
        }

        RCLCPP_ERROR(this->get_logger(), "Failed to call cart presence service");
        return false;
    }

    bool callDetectProducts(int32_t robot_id, int32_t order_id, const std::vector<int32_t>& expected_product_ids)
    {
        auto request = std::make_shared<DetectProductsInCart::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->expected_product_ids = expected_product_ids;

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
                "Detection success: %s, total detected: %d, message: %s",
                response->success ? "true" : "false",
                response->total_detected,
                response->message.c_str());
            return response->success;
        }

        RCLCPP_ERROR(this->get_logger(), "Failed to call detect products service");
        return false;
    }

    bool callMovePose(int32_t robot_id, int32_t order_id, const std::string& pose_type)
    {
        RCLCPP_INFO(this->get_logger(), "test1");
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
                         const std::string& arm_side, const std::vector<float>& target_position,
                         const std::vector<int>& bbox)
    {
        RCLCPP_INFO(this->get_logger(), "test2");
        auto request = std::make_shared<ArmPickProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->arm_side = arm_side;

        // DetectedProduct 생성
        request->target_product.product_id = product_id;
        request->target_product.confidence = 0.95f;

        if(target_position.size() == 6) {
            request->target_product.pose.x = target_position[0];
            request->target_product.pose.y = target_position[1];
            request->target_product.pose.z = target_position[2];
            request->target_product.pose.rx = target_position[3];
            request->target_product.pose.ry = target_position[4];
            request->target_product.pose.rz = target_position[5];
        }
        if(bbox.size() == 4) {
            request->target_product.bbox.x1 = bbox[0];
            request->target_product.bbox.y1 = bbox[1];
            request->target_product.bbox.x2 = bbox[2];
            request->target_product.bbox.y2 = bbox[3];
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
        RCLCPP_INFO(this->get_logger(), "test3");
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

private:
    rclcpp::Service<CheckAvailability>::SharedPtr check_avail_service_;
    rclcpp::Service<StartPacking>::SharedPtr start_packing_service_;

    rclcpp::Client<CheckCartPresence>::SharedPtr cart_presence_client_;
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
    auto node = std::make_shared<PackingServiceServer>();

    int32_t robot_id = 1;
    int32_t order_id = 123;

    if (!node->callCheckCartPresence(robot_id)) {
        RCLCPP_ERROR(node->get_logger(), "Cart not present. Abort.");
        rclcpp::shutdown();
        return 1;
    }

    std::vector<int32_t> expected_products = {101, 102, 103};
    if (!node->callDetectProducts(robot_id, order_id, expected_products)) {
        RCLCPP_ERROR(node->get_logger(), "Product detection failed. Abort.");
        rclcpp::shutdown();
        return 1;
    }

    if(!node->callMovePose(robot_id, order_id, "ready_pose")) {
        RCLCPP_ERROR(node->get_logger(), "MovePose failed. Abort.");
        rclcpp::shutdown();
        return 1;
    }

    if(!node->callPickProduct(robot_id, order_id, 101, "left", {0.5f,0.2f,0.1f}, {10,20,30,40})) {
        RCLCPP_ERROR(node->get_logger(), "PickProduct failed. Abort.");
        rclcpp::shutdown();
        return 1;
    }

    if(!node->callPlaceProduct(robot_id, order_id, 101, "left", {1.0f,0.5f,0.0f})) {
        RCLCPP_ERROR(node->get_logger(), "PlaceProduct failed. Abort.");
        rclcpp::shutdown();
        return 1;
    }

    if (node->callVerifyPacking(robot_id, order_id)) {
        RCLCPP_INFO(node->get_logger(), "Packing completed successfully.");
    } else {
        RCLCPP_WARN(node->get_logger(), "Packing not yet complete.");
    }

    rclcpp::shutdown();
    return 0;
}