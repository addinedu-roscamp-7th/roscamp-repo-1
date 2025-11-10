#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_start.hpp"
#include "shopee_interfaces/msg/packee_packing_complete.hpp"
#include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

using StartPacking = shopee_interfaces::srv::PackeePackingStart;
using PackingComplete = shopee_interfaces::msg::PackeePackingComplete;
using VerifyPackingComplete = shopee_interfaces::srv::PackeeVisionVerifyPackingComplete;
using RobotStatus = shopee_interfaces::msg::PackeeRobotStatus;

class PackeePackingServer : public rclcpp::Node
{
public:
    PackeePackingServer() : Node("packee_packing_server")
    {
        robot_id_ = 0;
        robot_status_ = "";

        reentrant_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::Reentrant);

        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = reentrant_cb_group_;

        start_packing_service_ = this->create_service<StartPacking>(
            "/packee/packing/start",
            std::bind(&PackeePackingServer::startPackingCallback, this, _1, _2),
            10,
            reentrant_cb_group_
        );

        robot_status_pub_ = this->create_publisher<RobotStatus>(
            "/packee/set_robot_status", 10);

        packing_complete_pub_ = this->create_publisher<PackingComplete>(
            "/packee/packing_complete", 10);

        verify_packing_client_ = this->create_client<VerifyPackingComplete>(
            "/packee/vision/verify_packing_complete",
            10,
            reentrant_cb_group_
        );

        robot_status_sub_ = this->create_subscription<RobotStatus>(
            "/packee/robot_status", 
            10,
            [this](RobotStatus::SharedPtr msg) { robot_status_ = msg->state; },
            sub_options
        );

        RCLCPP_INFO(this->get_logger(), "PackeePackingServer started (Reentrant + Async)");
    }

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

        std::vector<int32_t> sequence_list;
        for (auto &product_id : request->products)
            sequence_list.push_back(product_id);

        bool mtc_result = callMtcService(request->robot_id, request->order_id, sequence_list);

        if (!mtc_result) {
            RCLCPP_ERROR(this->get_logger(), "MTC service call failed");
            publishPackingComplete(request->robot_id, request->order_id, false, 0, "MTC service failed");
            publishRobotStatus(request->robot_id, "STANDBY", request->order_id, 0);
            return;
        }

        callVerifyPackingAsync(request->robot_id, request->order_id, request->products.size());

        RCLCPP_INFO(this->get_logger(), "포장 검증 요청 완료");
    }

    bool callMtcService(int32_t robot_id, int32_t order_id, const std::vector<int32_t> &sequence_list)
    {
        // TODO: 실제 MTC 로직 연결
        RCLCPP_INFO(this->get_logger(), "MTC 서비스 호출 완료");
        return true;
    }

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
                            "VerifyPacking result: cart_empty=%s, remaining=%d, msg=%s",
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

        RCLCPP_INFO(this->get_logger(), "VerifyPacking async request sent");
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