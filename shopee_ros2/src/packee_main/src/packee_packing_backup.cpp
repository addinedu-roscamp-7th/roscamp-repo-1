#include <chrono>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/srv/packee_packing_start.hpp"
#include "shopee_interfaces/msg/packee_packing_complete.hpp"
#include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"
#include "shopee_interfaces/msg/packee_robot_status.hpp"
#include "shopee_interfaces/srv/packee_vision_bpp_start.hpp"
#include "shopee_interfaces/srv/packee_main_start_mtc.hpp"
#include "shopee_interfaces/srv/packee_arm_packing_complete.hpp"

// 필요 시: #include "shopee_interfaces/msg/product_info.hpp"

using namespace std::chrono_literals;
using StartPacking          = shopee_interfaces::srv::PackeePackingStart;
using PackingCompleteMsg    = shopee_interfaces::msg::PackeePackingComplete;
using VerifyPackingComplete = shopee_interfaces::srv::PackeeVisionVerifyPackingComplete;
using RobotStatusMsg        = shopee_interfaces::msg::PackeeRobotStatus;
using BppStart              = shopee_interfaces::srv::PackeeVisionBppStart;
using StartMtc              = shopee_interfaces::srv::PackeeMainStartMtc;
using ArmPackingComplete    = shopee_interfaces::srv::PackeeArmPackingComplete;
using SequenceMsg           = shopee_interfaces::msg::Sequence;

class PackeePackingServer : public rclcpp::Node
{
public:
  PackeePackingServer() : Node("packee_packing_server"),
                          robot_id_(0), order_id_(0), robot_status_(""), phase_(Phase::IDLE)
  {
    // 콜백 그룹
    reentrant_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // ---- Subscriber ----
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = reentrant_cb_group_;

    robot_status_sub_ = this->create_subscription<RobotStatusMsg>(
      "/packee/robot_status",
      10,
      [this](RobotStatusMsg::SharedPtr msg) { robot_status_ = msg->state; },
      sub_options
    );

    // ---- Publishers ----
    robot_status_pub_ = this->create_publisher<RobotStatusMsg>("/packee/set_robot_status", 10);
    packing_complete_pub_ = this->create_publisher<PackingCompleteMsg>("/packee/packing_complete", 10);

    // ---- Services (Server) ----
    rclcpp::ServiceOptions srv_opt;
    srv_opt.callback_group = reentrant_cb_group_;

    start_packing_service_ = this->create_service<StartPacking>(
      "/packee/packing/start",
      std::bind(&PackeePackingServer::startPackingCallback, this, std::placeholders::_1, std::placeholders::_2),
      srv_opt
    );

    // BPP 완료는 vision 쪽에서 호출해 준다고 가정
    bpp_complete_server_ = this->create_service<BppStart>(
      "/packee/vision/bpp_complete",
      std::bind(&PackeePackingServer::bppCompleteCallback, this, std::placeholders::_1, std::placeholders::_2),
      srv_opt
    );

    mtc_complete_server_ = this->create_service<ArmPackingComplete>(
      "/packee/mtc/finish",
      std::bind(&PackeePackingServer::mtcCompleteCallback, this, std::placeholders::_1, std::placeholders::_2),
      srv_opt
    );

    // ---- Clients ----
    rclcpp::ClientOptions cli_opt;
    cli_opt.callback_group = reentrant_cb_group_;

    verify_packing_client_ = this->create_client<VerifyPackingComplete>("/packee/vision/verify_packing_complete", cli_opt);
    bpp_start_client_      = this->create_client<BppStart>("/packee/vision/bpp_start", cli_opt);
    start_mtc_client_      = this->create_client<StartMtc>("/packee/mtc/startmtc", cli_opt);

    RCLCPP_INFO(this->get_logger(), "PackeePackingServer started (Reentrant + Async)");
  }

private:
  // === 상태 머신 ===
  enum class Phase { IDLE, PACKING_STARTED, BPP_RUNNING, BPP_DONE, MTC_RUNNING, MTC_DONE, VERIFYING };

  // ---- Utilities ----
  void publishRobotStatus(int id, const std::string &state, int order_id, int items_in_cart)
  {
    RobotStatusMsg msg;
    msg.robot_id = id;
    msg.state = state;
    msg.current_order_id = order_id;
    msg.items_in_cart = items_in_cart;
    robot_status_pub_->publish(msg);
  }

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

  // ---- Service: /packee/packing/start ----
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

    // BPP 시작 (제품 정보 구조는 프로젝트 정의에 맞게 수정)
    // 예: request 안에 products 벡터가 있다고 가정하면 반복 호출 가능
    // for (const auto& p : request->products) { bppStartAsync(robot_id_, order_id_, p); }
    bppStartAsync(robot_id_, order_id_);
  }

  // ---- BPP 시작(Client) : 비동기 ----
  void bppStartAsync(int32_t robot_id, int32_t order_id /*, const ProductInfo& p*/)
  {
    auto req = std::make_shared<BppStart::Request>();
    req->robot_id = robot_id;
    req->order_id = order_id;
    // req->product_info = p; // 실제 필드에 맞게

    if (!bpp_start_client_->wait_for_service(1s)) {
      RCLCPP_ERROR(this->get_logger(), "BppStart service not available");
      return;
    }

    phase_ = Phase::BPP_RUNNING;

    bpp_start_client_->async_send_request(
      req,
      [this](rclcpp::Client<BppStart>::SharedFuture f) {
        auto res = f.get();
        RCLCPP_INFO(this->get_logger(), "BppStart result: success=%s, msg=%s",
                    res->success ? "true" : "false", res->message.c_str());
        // BPP 완료 자체는 vision이 /packee/vision/bpp_complete 로 알려준다고 가정 → 거기서 sequences_ 채움
      }
    );
  }

  // ---- BPP 완료(Server 콜백) : sequences_ 수신 ----
  bool bppCompleteCallback(const std::shared_ptr<BppStart::Request> request,
                           std::shared_ptr<BppStart::Response> response)
  {
    robot_id_ = request->robot_id;
    order_id_ = request->order_id;
    sequences_ = request->sequences;     // 벡터 복사 (필드명이 실제 srv에 맞는지 확인)
    RCLCPP_INFO(this->get_logger(), "BPP 완료: robot_id=%d, order_id=%d, sequence_size=%zu",
                robot_id_, order_id_, sequences_.size());

    response->success = true;
    response->message = "BPP complete received";

    // 높이 기준으로 시퀀스 정렬(예: z>0.175 먼저)
    std::vector<SequenceMsg> seq_sorted;
    for (const auto &s : sequences_) if (s.z > 0.175) seq_sorted.push_back(s);
    for (const auto &s : sequences_) if (s.z <= 0.175) seq_sorted.push_back(s);

    // MTC 시작
    mtcStartAsync(robot_id_, order_id_, seq_sorted);
    phase_ = Phase::BPP_DONE;
    return true;
  }

  // ---- MTC 시작(Client) : 비동기 ----
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
        RCLCPP_INFO(this->get_logger(), "StartMtc result: success=%s, msg=%s",
                    res->success ? "true" : "false", res->message.c_str());
        // 실제 MTC 종료는 /packee/mtc/finish 서버 콜백에서 받음
      }
    );
  }

  // ---- MTC 완료(Server 콜백) ----
  bool mtcCompleteCallback(const std::shared_ptr<ArmPackingComplete::Request> request,
                           std::shared_ptr<ArmPackingComplete::Response> response)
  {
    const bool success = request->success;
    RCLCPP_INFO(this->get_logger(), "MTC 완료: success=%s, msg=%s",
                success ? "true" : "false", request->message.c_str());

    response->success = true;
    response->message = "MTC complete received";
    phase_ = Phase::MTC_DONE;

    // 포장 검증 단계로 이동
    callVerifyPackingAsync(robot_id_, order_id_, /*product_count=*/static_cast<int32_t>(/* TODO: 실제 개수 */ 0));
    return true; // 반드시 bool 반환
  }

  // ---- Verify Packing (Client) : 비동기 ----
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
        const bool cart_empty = res->cart_empty;
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
      }
    );
  }

private:
  // 상태
  int robot_id_;
  int order_id_;
  std::string robot_status_;
  Phase phase_;
  std::vector<SequenceMsg> sequences_;

  // 콜백 그룹
  rclcpp::CallbackGroup::SharedPtr reentrant_cb_group_;

  // 통신 엔드포인트
  rclcpp::Service<StartPacking>::SharedPtr        start_packing_service_;
  rclcpp::Service<BppStart>::SharedPtr            bpp_complete_server_;
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