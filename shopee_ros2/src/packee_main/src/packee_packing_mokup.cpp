#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_start.hpp"
#include "shopee_interfaces/msg/packee_packing_complete.hpp"
#include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"
#include "shopee_interfaces/srv/packee_pick_product.hpp"
#include "shopee_interfaces/srv/packee_place_product.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

using StartPacking = shopee_interfaces::srv::PackeePackingStart;
using PackingComplete = shopee_interfaces::msg::PackeePackingComplete;
using VerifyPackingComplete = shopee_interfaces::srv::PackeeVisionVerifyPackingComplete;
using RobotStatus = shopee_interfaces::msg::PackeeRobotStatus;
using PickProduct = shopee_interfaces::srv::PackeePickProduct;
using PlaceProduct = shopee_interfaces::srv::PackeePlaceProduct;

class PackeePackingServer : public rclcpp::Node
{
public:
    PackeePackingServer() : Node("packee_packing_server")
    {
        robot_id_ = 0;
        robot_status_ = "";

        reentrant_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::Reentrant);

        // service server
        start_packing_service_ = this->create_service<StartPacking>(
            "/packee/packing/start",
            std::bind(&PackeePackingServer::startPackingCallback, this, _1, _2),
            rmw_qos_profile_services_default,
            reentrant_cb_group_
        );

        // publishers and subscribers
        robot_status_pub_ = this->create_publisher<RobotStatus>(
            "/packee/set_robot_status", rmw_qos_profile_topics_default);

        packing_complete_pub_ = this->create_publisher<PackingComplete>(
            "/packee/packing_complete", rmw_qos_profile_topics_default);

        robot_status_sub_ = this->create_subscription<RobotStatus>(
            "/packee/robot_status", 10,
            [this](RobotStatus::SharedPtr msg) { robot_status_ = msg->state; },
            reentrant_cb_group_
        );

        // service clients
        verify_packing_client_ = this->create_client<VerifyPackingComplete>(
            "/packee/vision/verify_packing_complete",
            rmw_qos_profile_services_default,
            reentrant_cb_group_
        );

        pick_product_right_client_ = this->create_client<PickProduct>(
            "/packee1/arm/pick_product", 
            rmw_qos_profile_services_default,
            reentrant_cb_group_
        );

        pick_product_left_client_ = this->create_client<PickProduct>(
            "/packee2/arm/pick_product", 
            rmw_qos_profile_services_default,
            reentrant_cb_group_
        );

        place_product_right_client_ = this->create_client<PlaceProduct>(
            "/packee1/mtc/place_product", 
            rmw_qos_profile_services_default,
            reentrant_cb_group_
        );

        place_product_left_client_ = this->create_client<PlaceProduct>(
            "/packee2/mtc/place_product", 
            rmw_qos_profile_services_default,
            reentrant_cb_group_
        );

        RCLCPP_INFO(this->get_logger(), "PackeePackingServer started");
    }

    // 로봇 상태 전송
    void publishRobotStatus(int id, const std::string &state, int order_id, int items_in_cart)
    {
        auto msg = RobotStatus();
        msg.robot_id = id;
        msg.state = state;
        msg.current_order_id = order_id;
        msg.items_in_cart = items_in_cart;

        robot_status_pub_->publish(msg);

        RCLCPP_INFO(this->get_logger(),
                    "RobotStatus published: robot_id=%d, state=%s, order_id=%d, items_in_cart=%d",
                    id, state.c_str(), order_id, items_in_cart);
    }

    // 포장 완료
    void publishPackingComplete(int32_t robot_id, int32_t order_id,
                                bool success, int32_t packed_items,
                                const std::string &message)
    {
        auto msg = PackingComplete();
        msg.robot_id = robot_id;
        msg.order_id = order_id;
        msg.success = success;
        msg.packed_items = packed_items;
        msg.message = message;

        packing_complete_pub_->publish(msg);

        RCLCPP_INFO(this->get_logger(),
                    "PackingComplete published: robot_id=%d, order_id=%d, success=%s, packed_items=%d, msg=%s",
                    robot_id, order_id, success ? "true" : "false", packed_items, message.c_str());
    }

    // 포장 시작
    void startPackingCallback(
        const std::shared_ptr<StartPacking::Request> request,
        std::shared_ptr<StartPacking::Response> response)
    {
        if (robot_status_ != "CHECKING_CART") {
            response->success = false;
            response->message = "Robot is not in CHECKING_CART state.";
            RCLCPP_WARN(this->get_logger(), "Robot not ready (status=%s)", robot_status_.c_str());
            return;
        }

        response->success = true;
        response->message = "Packing started";

        RCLCPP_INFO(this->get_logger(), "포장 시작 (order=%d, robot=%d)", request->order_id, request->robot_id);

        publishRobotStatus(request->robot_id, "PACKING_PRODUCT", request->order_id, 0);
        RCLCPP_INFO(this->get_logger(), "상품 pick & place 시퀀스 시작");

        int32_t index = 0;
        for (auto &product : request->products) {
            if (index % 2 == 0) {
                callRightPickProductService(request->robot_id, request->order_id, product.product_id, "right", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
                callRightPlaceProductService(request->robot_id, request->order_id, product.product_id, "right", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            } else {
                callLeftPickProductService(request->robot_id, request->order_id, product.product_id, "left", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
                callLeftPlaceProductService(request->robot_id, request->order_id, product.product_id, "left", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            }
            index++;
        }
            
        callVerifyPackingAsync(request->robot_id, request->order_id, request->products.size());
        RCLCPP_INFO(this->get_logger(), "포장 완료");
    }

    // packee1 pickup
    void callRightPickProductService(int32_t robot_id, int32_t order_id, int32_t product_id, std::string arm_side, std::vector<float> pose)
    {
        auto request = std::make_shared<PickProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;
        request->pose = pose;
        std::ostringstream pose_str;
        for (size_t i = 0; i < pose.size(); ++i) {
            pose_str << pose[i];
            if (i < pose.size() - 1) pose_str << ", ";
        }
        RCLCPP_INFO(this->get_logger(), "packee1 pickup request: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s, pose=[%s]",
                     robot_id, order_id, product_id, arm_side.c_str(), pose_str.str().c_str());

        if (!pick_product_right_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "packee1 pick product service not available");
            return;
        }

        pick_product_right_client_->async_send_request(
            request,
            [this, robot_id, order_id, product_id, arm_side, pose](rclcpp::Client<PickProduct>::SharedFuture future_response) {
                auto response = future_response.get();
                bool success = response->success;
                std::string msg = response->message;

                RCLCPP_INFO(this->get_logger(),
                            "Pick Product result: success=%s, msg=%s",
                            success ? "true" : "false",
                            msg.c_str());
            });
    }

    // packee2 pickup
    void callLeftPickProductService(int32_t robot_id, int32_t order_id, int32_t product_id, std::string arm_side, std::vector<float> pose)
    {
        auto request = std::make_shared<PickProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;
        request->pose = pose;
        std::ostringstream pose_str;
        for (size_t i = 0; i < pose.size(); ++i) {
            pose_str << pose[i];
            if (i < pose.size() - 1) pose_str << ", ";
        }
        RCLCPP_INFO(this->get_logger(), "packee2 pickup request: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s, pose=[%s]",
                     robot_id, order_id, product_id, arm_side.c_str(), pose_str.str().c_str());

        if (!pick_product_left_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "packee2 pick product service not available");
            return;
        }

        pick_product_left_client_->async_send_request(
            request,
            [this, robot_id, order_id, product_id, arm_side, pose](rclcpp::Client<PickProduct>::SharedFuture future_response) {
                auto response = future_response.get();
                bool success = response->success;
                std::string msg = response->message;

                RCLCPP_INFO(this->get_logger(),
                            "packee2 Pick Product result: success=%s, msg=%s",
                            success ? "true" : "false",
                            msg.c_str());
            });
    }

    // packee1 place
    void callRightPlaceProductService(int32_t robot_id, int32_t order_id, int32_t product_id, std::string arm_side, std::vector<float> pose)
    {
        auto request = std::make_shared<PlaceProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;
        request->pose = pose;
        std::ostringstream pose_str;
        for (size_t i = 0; i < pose.size(); ++i) {
            pose_str << pose[i];
            if (i < pose.size() - 1) pose_str << ", ";
        }
        RCLCPP_INFO(this->get_logger(), "packee1 place request: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s, pose=[%s]",
                     robot_id, order_id, product_id, arm_side.c_str(), pose_str.str().c_str());

        if (!place_product_right_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "packee2 place product service not available");
            return;
        }

        place_product_right_client_->async_send_request(
            request,
            [this, robot_id, order_id, product_id, arm_side, pose](rclcpp::Client<PlaceProduct>::SharedFuture future_response) {
                auto response = future_response.get();
                bool success = response->success;
                std::string msg = response->message;

                RCLCPP_INFO(this->get_logger(),
                            "packee1 place Product result: success=%s, msg=%s",
                            success ? "true" : "false",
                            msg.c_str());
            });
    }

    // packee2 place
    void callLeftPlaceProductService(int32_t robot_id, int32_t order_id, int32_t product_id, std::string arm_side, std::vector<float> pose)
    {
        auto request = std::make_shared<PlaceProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;
        request->pose = pose;
        std::ostringstream pose_str;
        for (size_t i = 0; i < pose.size(); ++i) {
            pose_str << pose[i];
            if (i < pose.size() - 1) pose_str << ", ";
        }
        RCLCPP_INFO(this->get_logger(), "packee2 place request: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s, pose=[%s]",
                     robot_id, order_id, product_id, arm_side.c_str(), pose_str.str().c_str());

        if (!place_product_left_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "packee2 place product service not available");
            return;
        }

        place_product_left_client_->async_send_request(
            request,
            [this, robot_id, order_id, product_id, arm_side, pose](rclcpp::Client<PlaceProduct>::SharedFuture future_response) {
                auto response = future_response.get();
                bool success = response->success;
                std::string msg = response->message;

                RCLCPP_INFO(this->get_logger(),
                            "packee2 Place Product result: success=%s, msg=%s",
                            success ? "true" : "false",
                            msg.c_str());
            });
    }

    // 포장 검증
    void callVerifyPackingAsync(int32_t robot_id, int32_t order_id, int32_t product_count)
    {
        auto request = std::make_shared<VerifyPackingComplete::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;

        if (!verify_packing_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "VerifyPacking service not available");
            publishPackingComplete(robot_id, order_id, false, 0, "VerifyPacking service unavailable");
            publishRobotStatus(robot_id, "STANDBY", order_id, 0);
            return;
        }

        verify_packing_client_->async_send_request(
            request,
            [this, robot_id, order_id, product_count](rclcpp::Client<VerifyPackingComplete>::SharedFuture future_response) {
                auto response = future_response.get();
                bool success = response->cart_empty;
                std::string msg = response->message;

                RCLCPP_INFO(this->get_logger(),
                            "📸 VerifyPacking result: cart_empty=%s, remaining=%d, msg=%s",
                            success ? "true" : "false",
                            response->remaining_items,
                            msg.c_str());

                if (success)
                    publishPackingComplete(robot_id, order_id, true, product_count, "Success Packing");
                else
                    publishPackingComplete(robot_id, order_id, false, product_count, "Failed Packing");

                publishRobotStatus(robot_id, "STANDBY", order_id, 0);
                RCLCPP_INFO(this->get_logger(), "포장 완료, 대기 상태 복귀");
            });
    }

private:
    int robot_id_;
    std::string robot_status_;

    rclcpp::CallbackGroup::SharedPtr reentrant_cb_group_;

    rclcpp::Service<StartPacking>::SharedPtr start_packing_service_;
    rclcpp::Publisher<RobotStatus>::SharedPtr robot_status_pub_;
    rclcpp::Subscription<RobotStatus>::SharedPtr robot_status_sub_;
    rclcpp::Publisher<PackingComplete>::SharedPtr packing_complete_pub_;
    rclcpp::Client<VerifyPackingComplete>::SharedPtr verify_packing_client_;
    rclcpp::Client<PickProduct>::SharedPtr pick_product_right_client_;
    rclcpp::Client<PickProduct>::SharedPtr pick_product_left_client_;
    rclcpp::Client<PlaceProduct>::SharedPtr place_product_right_client_;
    rclcpp::Client<PlaceProduct>::SharedPtr place_product_left_client_;

};

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