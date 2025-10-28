#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/msg/b_box.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/arm_task_status.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/arm_place_product.hpp"

using namespace std::chrono_literals;

class MockPackeeMain : public rclcpp::Node
{
public:
  MockPackeeMain()
  : Node("mock_packee_main"), state_("wait_services"), current_arm_index_(0)
  {
    this->declare_parameter<int>("robot_id", 1);
    this->declare_parameter<int>("order_id", 100);
    this->declare_parameter<int>("product_id", 501);
    this->declare_parameter<std::string>("arm_side", "left");
    this->declare_parameter<std::string>("arm_sides", "left,right");

    robot_id_ = this->get_parameter("robot_id").as_int();
    order_id_ = this->get_parameter("order_id").as_int();
    base_product_id_ = this->get_parameter("product_id").as_int();
    default_arm_side_ = this->get_parameter("arm_side").as_string();
    ParseArmSides(this->get_parameter("arm_sides").as_string());

    move_cli_ = this->create_client<shopee_interfaces::srv::ArmMoveToPose>("/packee/arm/move_to_pose");
    pick_cli_ = this->create_client<shopee_interfaces::srv::ArmPickProduct>("/packee/arm/pick_product");
    place_cli_ = this->create_client<shopee_interfaces::srv::ArmPlaceProduct>("/packee/arm/place_product");

    pose_sub_ = this->create_subscription<shopee_interfaces::msg::ArmPoseStatus>(
      "/packee/arm/pose_status", 10, std::bind(&MockPackeeMain::OnPoseStatus, this, std::placeholders::_1));
    pick_sub_ = this->create_subscription<shopee_interfaces::msg::ArmTaskStatus>(
      "/packee/arm/pick_status", 10, std::bind(&MockPackeeMain::OnPickStatus, this, std::placeholders::_1));
    place_sub_ = this->create_subscription<shopee_interfaces::msg::ArmTaskStatus>(
      "/packee/arm/place_status", 10, std::bind(&MockPackeeMain::OnPlaceStatus, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(200ms, std::bind(&MockPackeeMain::ProcessSteps, this));
    RCLCPP_INFO(this->get_logger(), "✅ MockPackeeMain 초기화 완료");
  }

private:
  using MoveCli = rclcpp::Client<shopee_interfaces::srv::ArmMoveToPose>;
  using PickCli = rclcpp::Client<shopee_interfaces::srv::ArmPickProduct>;
  using PlaceCli = rclcpp::Client<shopee_interfaces::srv::ArmPlaceProduct>;
  using MoveFuture = MoveCli::SharedFuture;
  using PickFuture = PickCli::SharedFuture;
  using PlaceFuture = PlaceCli::SharedFuture;
  std::variant<std::monostate, MoveFuture, PickFuture, PlaceFuture> current_future_;

  void ProcessSteps()
  {
    if (state_ == "wait_services")
    {
      if (move_cli_->service_is_ready() && pick_cli_->service_is_ready() && place_cli_->service_is_ready())
      {
        RCLCPP_INFO(this->get_logger(), "서비스 준비 완료 → Move 요청");
        SendMoveRequest();
        state_ = "await_move";
      }
    }
    else if (state_ == "await_move")
      HandleFuture("자세 변경");
    else if (state_ == "request_pick")
    {
      if (current_arm_index_ >= arm_sides_.size())
        state_ = "completed";
      else
      {
        SendPickRequest();
        state_ = "await_pick";
      }
    }
    else if (state_ == "await_pick")
      HandleFuture("상품 픽업");
    else if (state_ == "request_place")
    {
      SendPlaceRequest();
      state_ = "await_place";
    }
    else if (state_ == "await_place")
      HandleFuture("상품 담기");
    else if (state_ == "completed")
    {
      RCLCPP_INFO(this->get_logger(), "✅ 테스트 완료. 노드 종료");
      rclcpp::shutdown();
    }
  }

  void HandleFuture(const std::string &action)
  {
    bool ready = false, ok = true;
    std::string msg;

    std::visit([&](auto &f) {
      using F = std::decay_t<decltype(f)>;
      if constexpr (!std::is_same_v<F, std::monostate>)
      {
        if (!f.valid()) return;
        if (f.wait_for(0s) != std::future_status::ready) return;
        auto res = f.get();
        ok = res->success;
        msg = res->message;
        ready = true;
      }
    }, current_future_);

    if (!ready) return;
    current_future_ = std::monostate{};

    if (!ok)
    {
      RCLCPP_ERROR(this->get_logger(), "❌ %s 실패: %s", action.c_str(), msg.c_str());
      state_ = "completed";
      return;
    }

    RCLCPP_INFO(this->get_logger(), "✅ %s 성공: %s", action.c_str(), msg.c_str());
    if (action == "자세 변경") state_ = "request_pick";
    else if (action == "상품 픽업") state_ = "request_place";
    else if (action == "상품 담기")
    {
      ++current_arm_index_;
      if (current_arm_index_ < arm_sides_.size())
      {
        RCLCPP_INFO(this->get_logger(), "➡️ 다음 팔 테스트: %s (product_id=%d)",
                    CurrentArmSide().c_str(), CurrentProductId());
        state_ = "request_pick";
      }
      else state_ = "completed";
    }
  }

  // ---------- 요청 ----------
  void SendMoveRequest()
  {
    auto req = std::make_shared<shopee_interfaces::srv::ArmMoveToPose::Request>();
    req->robot_id = robot_id_;
    req->order_id = order_id_;
    req->pose_type = "cart_view";
    current_future_ = move_cli_->async_send_request(req);
  }

  void SendPickRequest()
  {
    auto req = std::make_shared<shopee_interfaces::srv::ArmPickProduct::Request>();
    req->robot_id = robot_id_;
    req->order_id = order_id_;
    req->arm_side = CurrentArmSide();
    req->target_product.product_id = CurrentProductId();

    AssignDetectedProductPose(&req->target_product, 0.25F, 0.0F, 0.12F, 0.0F, 0.92F);
    req->target_product.bbox = CreateBBox(120, 180, 250, 320);
    current_future_ = pick_client_->async_send_request(req);


    req->target_product.pose.joint_1 = 0.25F;
    req->target_product.pose.joint_2 = 0.0F;
    req->target_product.pose.joint_3 = 0.12F;
    req->target_product.pose.joint_4 = 0.0F;  // optional

    req->target_product.bbox.x1 = 120;
    req->target_product.bbox.y1 = 180;
    req->target_product.bbox.x2 = 250;
    req->target_product.bbox.y2 = 320;

    current_future_ = pick_cli_->async_send_request(req);
  }

  void SendPlaceRequest()
  {
    auto req = std::make_shared<shopee_interfaces::srv::ArmPlaceProduct::Request>();
    req->robot_id = robot_id_;
    req->order_id = order_id_;
    req->product_id = CurrentProductId();
    req->arm_side = CurrentArmSide();

    // 🟢 [FIXED] 기존 req->target_pose → req->pose 로 수정
    req->pose.joint_1 = 0.35F;
    req->pose.joint_2 = 0.10F;
    req->pose.joint_3 = 0.15F;
    req->pose.joint_4 = 0.0F;

    current_future_ = place_cli_->async_send_request(req);
  }

  // ---------- 콜백 ----------
  void OnPoseStatus(const shopee_interfaces::msg::ArmPoseStatus::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "[Pose] %s: %s (%.2f) %s",
                msg->pose_type.c_str(), msg->status.c_str(),
                msg->progress, msg->message.c_str());
  }
  void OnPickStatus(const shopee_interfaces::msg::ArmTaskStatus::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "[Pick] %d %s %s (%.2f) %s",
                msg->product_id, msg->current_phase.c_str(),
                msg->status.c_str(), msg->progress, msg->message.c_str());
  }
  void OnPlaceStatus(const shopee_interfaces::msg::ArmTaskStatus::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "[Place] %d %s %s (%.2f) %s",
                msg->product_id, msg->current_phase.c_str(),
                msg->status.c_str(), msg->progress, msg->message.c_str());
  }

  // ---------- 유틸 ----------
  void ParseArmSides(const std::string &raw)
  {
    std::string tmp;
    for (char c : raw)
    {
      if (c == ',')
      {
        if (!tmp.empty()) arm_sides_.push_back(tmp), tmp.clear();
      }
      else tmp += c;
    }
    if (!tmp.empty()) arm_sides_.push_back(tmp);
  }

  std::string CurrentArmSide() const
  {
    return (current_arm_index_ < arm_sides_.size()) ? arm_sides_[current_arm_index_] : default_arm_side_;
  }

  int32_t CurrentProductId() const { return base_product_id_ + (int32_t)current_arm_index_; }

  // ---------- 멤버 ----------
  int robot_id_, order_id_, base_product_id_;
  std::string default_arm_side_;
  std::vector<std::string> arm_sides_;
  std::string state_;
  size_t current_arm_index_;

  rclcpp::TimerBase::SharedPtr timer_;
  MoveCli::SharedPtr move_cli_;
  PickCli::SharedPtr pick_cli_;
  PlaceCli::SharedPtr place_cli_;
  rclcpp::Subscription<shopee_interfaces::msg::ArmPoseStatus>::SharedPtr pose_sub_;
  rclcpp::Subscription<shopee_interfaces::msg::ArmTaskStatus>::SharedPtr pick_sub_;
  rclcpp::Subscription<shopee_interfaces::msg::ArmTaskStatus>::SharedPtr place_sub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MockPackeeMain>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
