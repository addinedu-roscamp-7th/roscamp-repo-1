#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_start.hpp"
#include "shopee_interfaces/msg/packee_packing_complete.hpp"
#include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"
#include "shopee_interfaces/srv/packee_vision_bpp_start.hpp"
#include "shopee_interfaces/srv/packee_vision_bpp_complete.hpp"
#include "shopee_interfaces/srv/packee_main_start_mtc.hpp"
#include "shopee_interfaces/srv/packee_arm_packing_complete.hpp"
#include "shopee_interfaces/msg/sequence.hpp"
#include "shopee_interfaces/msg/product_info.hpp"

using namespace std::chrono_literals;
using StartPacking          = shopee_interfaces::srv::PackeePackingStart;
using PackingCompleteMsg    = shopee_interfaces::msg::PackeePackingComplete;
using VerifyPackingComplete = shopee_interfaces::srv::PackeeVisionVerifyPackingComplete;
using RobotStatusMsg        = shopee_interfaces::msg::PackeeRobotStatus;
using BppStart              = shopee_interfaces::srv::PackeeVisionBppStart;
using BppComplete           = shopee_interfaces::srv::VisionCheckCartPresence;
using StartMtc              = shopee_interfaces::srv::PackeeMainStartMtc;
using ArmPackingComplete    = shopee_interfaces::srv::PackeeArmPackingComplete;
using SequenceMsg           = shopee_interfaces::msg::Sequence;
using ProductInfo           = shopee_interfaces::msg::ProductInfo;

class PackeePackingServer : public rclcpp::Node
{
public:
  PackeePackingServer() : Node("packee_packing_server"),
                          robot_id_(0), order_id_(0), robot_status_(""), phase_(Phase::IDLE)
  {
    reentrant_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = reentrant_cb_group_;

    robot_status_sub_ = this->create_subscription<RobotStatusMsg>(
      "/packee/robot_status",
      10,
      [this](RobotStatusMsg::SharedPtr msg) { robot_status_ = msg->state; },
      sub_options
    );

    robot_status_pub_ = this->create_publisher<RobotStatusMsg>("/packee/set_robot_status", 10);
    packing_complete_pub_ = this->create_publisher<PackingCompleteMsg>("/packee/packing_complete", 10);

    rclcpp::ServiceOptions srv_opt;
    srv_opt.callback_group = reentrant_cb_group_;

    start_packing_service_ = this->create_service<StartPacking>(
      "/packee/packing/start",
      std::bind(&PackeePackingServer::startPackingCallback, this, std::placeholders::_1, std::placeholders::_2),
      srv_opt
    );

    bpp_complete_server_ = this->create_service<BppComplete>(
      "/packee/vision/bpp_complete",
      std::bind(&PackeePackingServer::bppCompleteCallback, this, std::placeholders::_1, std::placeholders::_2),
      srv_opt
    );

    mtc_complete_server_ = this->create_service<ArmPackingComplete>(
      "/packee/mtc/finish",
      std::bind(&PackeePackingServer::mtcCompleteCallback, this, std::placeholders::_1, std::placeholders::_2),
      srv_opt
    );

    rclcpp::ClientOptions cli_opt;
    cli_opt.callback_group = reentrant_cb_group_;

    verify_packing_client_ = this->create_client<VerifyPackingComplete>("/packee/vision/verify_packing_complete", cli_opt);
    bpp_start_client_      = this->create_client<BppStart>("/packee/vision/bpp_start", cli_opt);
    start_mtc_client_      = this->create_client<StartMtc>("/packee/mtc/startmtc", cli_opt);

    RCLCPP_INFO(this->get_logger(), "PackeePackingServer started");
  }

private:
  // === 상태 머신 ===
  enum class Phase { IDLE, PACKING_STARTED, BPP_RUNNING, BPP_DONE, MTC_RUNNING, MTC_DONE, VERIFYING };

  // --- Robot Status Publish ---
  void publishRobotStatus(int id, const std::string &state, int order_id, int items_in_cart)
  {
    RobotStatusMsg msg;
    msg.robot_id = id;
    msg.state = state;
    msg.current_order_id = order_id;
    msg.items_in_cart = items_in_cart;
    robot_status_pub_->publish(msg);
  }

  // --- Packing Complete Publish ---
  void publishPackingComplete(int32_t robot_id, int32_t order_id,
                              bool success, int32_t packed_items,
                              const std::string &message)
  {
    PackingCompleteMsg msg;
    msg.robot_id = robot_id;
    msg.order_id = order_id;
    msg.success = success;
    msg.packed_items = packed_items;
    msg.message = message;
    packing_complete_pub_->publish(msg);
  }

  // === START PACKING ===
  void startPackingCallback(const std::shared_ptr<StartPacking::Request> request,
                            std::shared_ptr<StartPacking::Response> response)
  {
    if (robot_status_ != "CHECKING_CART") {
      response->success = false;
      response->message = "Robot is not in CHECKING_CART state.";
      RCLCPP_WARN(this->get_logger(), "Robot not ready (status=%s)", robot_status_.c_str());
      return;
    }

    robot_id_ = request->robot_id;
    order_id_ = request->order_id;
    sequences_.clear();

    response->success = true;
    response->message = "Packing started";
    phase_ = Phase::PACKING_STARTED;

    publishRobotStatus(robot_id_, "PACKING_PRODUCT", order_id_, 0);
    RCLCPP_INFO(this->get_logger(), "포장 시작 (order=%d, robot=%d)", order_id_, robot_id_);

    std::vector<ProductInfo> full_specs;
    for (const auto &p : request->products) {
      auto spec = getProductSpecById(p.product_id);
      full_specs.push_back(spec);
      RCLCPP_INFO(this->get_logger(),
        "  ▶ Product id=%d, size=(%d,%d,%d), weight=%d, fragile=%s",
        spec.product_id, spec.length, spec.width, spec.height,
        spec.weight, spec.fragile ? "true" : "false");
    }

    // BPP 시작
    bppStartAsync(robot_id_, order_id_, full_specs);
  }

  void bppStartAsync(int32_t robot_id, int32_t order_id, const std::vector<ProductInfo> &products)
  {
    auto req = std::make_shared<BppStart::Request>();
    req->robot_id = robot_id;
    req->order_id = order_id;
    req->products = products;

    if (!bpp_start_client_->wait_for_service(1s)) {
      RCLCPP_ERROR(this->get_logger(), "BppStart service not available");
      return;
    }

    phase_ = Phase::BPP_RUNNING;

    bpp_start_client_->async_send_request(
      req,
      [this](rclcpp::Client<BppStart>::SharedFuture f) {
        auto res = f.get();
        RCLCPP_INFO(this->get_logger(),
                    "BppStart result: success=%s, msg=%s",
                    res->success ? "true" : "false", res->message.c_str());
      });
  }

  bool bppCompleteCallback(const std::shared_ptr<BppComplete::Request> request,
                           std::shared_ptr<BppComplete::Response> response)
  {
    robot_id_ = request->robot_id;
    order_id_ = request->order_id;
    sequences_ = request->sequences;

    RCLCPP_INFO(this->get_logger(), "BPP 완료: robot_id=%d, order_id=%d, sequence_size=%zu",
                robot_id_, order_id_, sequences_.size());

    response->success = true;
    response->message = "BPP complete received";

    std::vector<SequenceMsg> seq_sorted;
    for (const auto &s : sequences_) if (s.x > 0.175) seq_sorted.push_back(s);
    for (const auto &s : sequences_) if (s.x < 0.175) seq_sorted.push_back(s);

    mtcStartAsync(robot_id_, order_id_, seq_sorted);
    phase_ = Phase::BPP_DONE;
    return true;
  }

  void mtcStartAsync(int32_t robot_id, int32_t order_id, const std::vector<SequenceMsg> &seqs)
  {
    auto req = std::make_shared<StartMtc::Request>();
    req->robot_id = robot_id;
    req->order_id = order_id;
    req->sequence_list = seqs;

    if (!start_mtc_client_->wait_for_service(1s)) {
      RCLCPP_ERROR(this->get_logger(), "StartMtc service not available");
      return;
    }

    phase_ = Phase::MTC_RUNNING;

    start_mtc_client_->async_send_request(
      req,
      [this](rclcpp::Client<StartMtc>::SharedFuture f) {
        auto res = f.get();
        RCLCPP_INFO(this->get_logger(),
                    "MTC start result: success=%s, msg=%s",
                    res->success ? "true" : "false", res->message.c_str());
      });
  }

  // === MTC COMPLETE ===
  bool mtcCompleteCallback(const std::shared_ptr<ArmPackingComplete::Request> request,
                           std::shared_ptr<ArmPackingComplete::Response> response)
  {
    bool success = request->success;
    RCLCPP_INFO(this->get_logger(), "MTC 완료: success=%s, msg=%s",
                success ? "true" : "false", request->message.c_str());

    response->success = true;
    response->message = "MTC complete received";
    phase_ = Phase::MTC_DONE;

    callVerifyPackingAsync(robot_id_, order_id_, static_cast<int32_t>(sequences_.size()));
    return true;
  }

  void callVerifyPackingAsync(int32_t robot_id, int32_t order_id, int32_t product_count)
  {
    auto req = std::make_shared<VerifyPackingComplete::Request>();
    req->robot_id = robot_id;
    req->order_id = order_id;

    if (!verify_packing_client_->wait_for_service(1s)) {
      RCLCPP_ERROR(this->get_logger(), "VerifyPacking service not available");
      publishPackingComplete(robot_id, order_id, false, 0, "VerifyPacking service unavailable");
      publishRobotStatus(robot_id, "STANDBY", order_id, 0);
      return;
    }

    phase_ = Phase::VERIFYING;

    verify_packing_client_->async_send_request(
      req,
      [this, robot_id, order_id, product_count](rclcpp::Client<VerifyPackingComplete>::SharedFuture f) {
        auto res = f.get();
        bool cart_empty = res->cart_empty;

        RCLCPP_INFO(this->get_logger(),
                    "VerifyPacking result: cart_empty=%s, remaining=%d, msg=%s",
                    cart_empty ? "true" : "false", res->remaining_items, res->message.c_str());

        if (cart_empty)
          publishPackingComplete(robot_id, order_id, true, product_count, "Success Packing");
        else
          publishPackingComplete(robot_id, order_id, false, product_count, "Failed Packing");

        publishRobotStatus(robot_id, "STANDBY", order_id, 0);
        RCLCPP_INFO(this->get_logger(), "포장 완료, 대기 상태 복귀");
        phase_ = Phase::IDLE;
      });
  }

  ProductInfo getProductSpecById(int32_t product_id)
  {
    ProductInfo info{};
    info.product_id = product_id;
    info.quantity = 1;
    info.length = 0;
    info.width = 0;
    info.height = 0;
    info.weight = 0;
    info.fragile = false;

    switch (product_id)
    {
      case 1: // Wasabi
        info.quantity = 2;
        info.length = 143;
        info.width = 27;
        info.height = 40;
        info.weight = 150;
        info.fragile = false;
        break;

      case 12: // Fish
        info.quantity = 2;
        info.length = 1700;
        info.width = 50;
        info.height = 450;
        info.weight = 100;
        info.fragile = false;
        break;

      case 14: // Eclipse
        info.quantity = 2;
        info.length = 76;
        info.width = 20;
        info.height = 40;
        info.weight = 520;
        info.fragile = false;
        break;

      default:
        break;
    }
    return info;
  }

private:
  int robot_id_;
  int order_id_;
  std::string robot_status_;
  Phase phase_;
  std::vector<SequenceMsg> sequences_;

  rclcpp::CallbackGroup::SharedPtr reentrant_cb_group_;

  // 통신 엔드포인트
  rclcpp::Service<StartPacking>::SharedPtr        start_packing_service_;
  rclcpp::Service<BppComplete>::SharedPtr         bpp_complete_server_;
  rclcpp::Service<ArmPackingComplete>::SharedPtr  mtc_complete_server_;
  rclcpp::Client<VerifyPackingComplete>::SharedPtr verify_packing_client_;
  rclcpp::Client<BppStart>::SharedPtr              bpp_start_client_;
  rclcpp::Client<StartMtc>::SharedPtr              start_mtc_client_;
  rclcpp::Publisher<RobotStatusMsg>::SharedPtr     robot_status_pub_;
  rclcpp::Subscription<RobotStatusMsg>::SharedPtr  robot_status_sub_;
  rclcpp::Publisher<PackingCompleteMsg>::SharedPtr packing_complete_pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PackeePackingServer>();
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}