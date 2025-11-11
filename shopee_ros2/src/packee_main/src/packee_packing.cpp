#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_start.hpp"
#include "shopee_interfaces/msg/packee_packing_complete.hpp"
#include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/arm_place_product.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

using StartPacking = shopee_interfaces::srv::PackeePackingStart;
using PackingComplete = shopee_interfaces::msg::PackeePackingComplete;
using VerifyPackingComplete = shopee_interfaces::srv::PackeeVisionVerifyPackingComplete;
using RobotStatus = shopee_interfaces::msg::PackeeRobotStatus;
using PickProduct = shopee_interfaces::srv::ArmPickProduct;
using PlaceProduct = shopee_interfaces::srv::ArmPlaceProduct;

class PackeePackingServer : public rclcpp::Node
{
public:
    PackeePackingServer() : Node("packee_packing_server")
    {
        robot_id_ = 0;
        robot_status_ = "";
        current_product_index_ = 0;
        total_product_count_ = 0;

        reentrant_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::Reentrant);

        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = reentrant_cb_group_;

        // service server
        start_packing_service_ = this->create_service<StartPacking>(
            "/packee/packing/start",
            std::bind(&PackeePackingServer::startPackingCallback, this, _1, _2),
            rclcpp::QoS(10),
            reentrant_cb_group_
        );

        // publishers and subscribers
        robot_status_pub_ = this->create_publisher<RobotStatus>("/packee/set_robot_status", rclcpp::QoS(10));
        packing_complete_pub_ = this->create_publisher<PackingComplete>("/packee/packing_complete", rclcpp::QoS(10));

        robot_status_sub_ = this->create_subscription<RobotStatus>(
            "/packee/robot_status",
            rclcpp::QoS(10),
            [this](RobotStatus::SharedPtr msg) { robot_status_ = msg->state; },
            sub_options
        );

        // service clients
        verify_packing_client_ = this->create_client<VerifyPackingComplete>(
            "/packee/vision/verify_packing_complete",
            rclcpp::QoS(10),
            reentrant_cb_group_
        );

        pick_product_right_client_ = this->create_client<PickProduct>(
            "/packee1/arm/pick_product",
            rclcpp::QoS(10),
            reentrant_cb_group_
        );

        pick_product_left_client_ = this->create_client<PickProduct>(
            "/packee2/arm/pick_product",
            rclcpp::QoS(10),
            reentrant_cb_group_
        );

        place_product_right_client_ = this->create_client<PlaceProduct>(
            "/packee1/arm/place_product",
            rclcpp::QoS(10),
            reentrant_cb_group_
        );

        place_product_left_client_ = this->create_client<PlaceProduct>(
            "/packee2/arm/place_product",
            rclcpp::QoS(10),
            reentrant_cb_group_
        );

        RCLCPP_INFO(this->get_logger(), "PackeePackingServer started");
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

        RCLCPP_INFO(this->get_logger(), "🧾 포장 시작 (order=%d, robot=%d)", request->order_id, request->robot_id);

        // 작업 정보 저장
        current_robot_id_ = request->robot_id;
        current_order_id_ = request->order_id;
        product_list_ = request->products;
        current_product_index_ = 0;
        total_product_count_ = static_cast<int32_t>(request->products.size());

        publishRobotStatus(request->robot_id, "PACKING_PRODUCT", request->order_id, 0);
        RCLCPP_INFO(this->get_logger(), "상품 pick & place 시퀀스 시작 (총 %d개)", total_product_count_);

        // 첫 번째 상품부터 순차 처리 시작
        processNextProduct();
    }

    static inline void setPoseFromVector(shopee_interfaces::msg::Pose6D &msg, const std::vector<float> &pose)
    {
        if (pose.size() < 6) {
            RCLCPP_WARN(rclcpp::get_logger("PackeePackingServer"),
                        "Pose vector size < 6 (size=%zu). Filling zeros.", pose.size());
            msg.x = msg.y = msg.z = msg.rx = msg.ry = msg.rz = 0.0f;
            return;
        }
        msg.x = pose[0];
        msg.y = pose[1];
        msg.z = pose[2];
        msg.rx = pose[3];
        msg.ry = pose[4];
        msg.rz = pose[5];
    }

    // 다음 상품 처리
    void processNextProduct()
    {
        if (current_product_index_ >= total_product_count_) {
            RCLCPP_INFO(this->get_logger(), "모든 상품 처리 완료. 검증 시작");
            callVerifyPackingAsync(current_robot_id_, current_order_id_, total_product_count_);
            return;
        }

        int32_t product_id = product_list_[current_product_index_];
        std::vector<float> pose = {0, 0, 0, 0, 0, 0};

        RCLCPP_INFO(this->get_logger(), 
                    "📦 상품 처리 [%d/%d]: product_id=%d", 
                    current_product_index_ + 1, total_product_count_, product_id);

        // 짝수: 오른팔, 홀수: 왼팔
        if (current_product_index_ % 2 == 0) {
            callRightPickProductService(current_robot_id_, current_order_id_, product_id, "right", pose);
        } else {
            callLeftPickProductService(current_robot_id_, current_order_id_, product_id, "left", pose);
        }
    }

    void callRightPickProductService(int32_t robot_id, int32_t order_id, int32_t product_id,
                                     const std::string &arm_side, const std::vector<float> &pose)
    {
        auto request = std::make_shared<PickProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;
        setPoseFromVector(request->pose, pose);

        RCLCPP_INFO(this->get_logger(),
                    "packee1 pickup request: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s",
                    robot_id, order_id, product_id, arm_side.c_str());

        if (!pick_product_right_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "packee1 pick product service not available");
            return;
        }

        pick_product_right_client_->async_send_request(
            request,
            [this, robot_id, order_id, product_id, arm_side, pose](rclcpp::Client<PickProduct>::SharedFuture future_response) {
                auto response = future_response.get();
                if (response->success) {
                    RCLCPP_INFO(this->get_logger(),
                                "✅ packee1 Pick 성공 (product_id=%d). Place 시작!", product_id);
                    this->callRightPlaceProductService(robot_id, order_id, product_id, arm_side, pose);
                } else {
                    RCLCPP_ERROR(this->get_logger(),
                                "❌ packee1 Pick 실패 (product_id=%d): %s",
                                product_id, response->message.c_str());
                    // 실패해도 다음 상품 진행
                    this->current_product_index_++;
                    this->processNextProduct();
                }
            });
    }

    void callLeftPickProductService(int32_t robot_id, int32_t order_id, int32_t product_id,
                                    const std::string &arm_side, const std::vector<float> &pose)
    {
        auto request = std::make_shared<PickProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;
        setPoseFromVector(request->pose, pose);

        RCLCPP_INFO(this->get_logger(),
                    "packee2 pickup request: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s",
                    robot_id, order_id, product_id, arm_side.c_str());

        if (!pick_product_left_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "packee2 pick product service not available");
            return;
        }

        pick_product_left_client_->async_send_request(
            request,
            [this, robot_id, order_id, product_id, arm_side, pose](rclcpp::Client<PickProduct>::SharedFuture future_response) {
                const auto response = future_response.get();
                if (response->success) {
                    RCLCPP_INFO(this->get_logger(),
                                "✅ packee2 Pick 성공 (product_id=%d). Place 시작!", product_id);
                    this->callLeftPlaceProductService(robot_id, order_id, product_id, arm_side, pose);
                } else {
                    RCLCPP_ERROR(this->get_logger(),
                                "❌ packee2 Pick 실패 (product_id=%d): %s",
                                product_id, response->message.c_str());
                    // 실패해도 다음 상품 진행
                    this->current_product_index_++;
                    this->processNextProduct();
                }
            });
    }

    void callRightPlaceProductService(int32_t robot_id, int32_t order_id, int32_t product_id,
                                      const std::string &arm_side, const std::vector<float> &pose)
    {
        auto request = std::make_shared<PlaceProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;
        setPoseFromVector(request->pose, pose);

        if (!place_product_right_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "packee1 place product service not available");
            return;
        }

        place_product_right_client_->async_send_request(
            request,
            [this, product_id](rclcpp::Client<PlaceProduct>::SharedFuture future_response) {
                const auto response = future_response.get();
                RCLCPP_INFO(this->get_logger(),
                            "packee1 Place Product result: success=%s, msg=%s",
                            response->success ? "true" : "false", response->message.c_str());
                
                // Place 완료 후 다음 상품으로
                this->current_product_index_++;
                this->processNextProduct();
            });
    }

    void callLeftPlaceProductService(int32_t robot_id, int32_t order_id, int32_t product_id,
                                     const std::string &arm_side, const std::vector<float> &pose)
    {
        auto request = std::make_shared<PlaceProduct::Request>();
        request->robot_id = robot_id;
        request->order_id = order_id;
        request->product_id = product_id;
        request->arm_side = arm_side;
        setPoseFromVector(request->pose, pose);

        if (!place_product_left_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(this->get_logger(), "packee2 place product service not available");
            return;
        }

        place_product_left_client_->async_send_request(
            request,
            [this, product_id](rclcpp::Client<PlaceProduct>::SharedFuture future_response) {
                const auto response = future_response.get();
                RCLCPP_INFO(this->get_logger(),
                            "packee2 Place Product result: success=%s, msg=%s",
                            response->success ? "true" : "false", response->message.c_str());
                
                // Place 완료 후 다음 상품으로
                this->current_product_index_++;
                this->processNextProduct();
            });
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
                const auto response = future_response.get();
                const bool empty = response->cart_empty;

                RCLCPP_INFO(this->get_logger(),
                            "📸 VerifyPacking result: cart_empty=%s, remaining=%d, msg=%s",
                            empty ? "true" : "false",
                            response->remaining_items,
                            response->message.c_str());

                if (empty)
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

    // 순차 처리를 위한 변수들
    int32_t current_robot_id_;
    int32_t current_order_id_;
    std::vector<int32_t> product_list_;
    int32_t current_product_index_;
    int32_t total_product_count_;

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