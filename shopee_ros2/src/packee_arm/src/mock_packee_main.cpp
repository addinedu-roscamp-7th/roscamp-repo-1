#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/arm_task_status.hpp"
#include "shopee_interfaces/msg/detected_product.hpp"
#include "shopee_interfaces/msg/pose6_d.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/arm_place_product.hpp"

using namespace std::chrono_literals;
using shopee_interfaces::msg::ArmPoseStatus;
using shopee_interfaces::msg::ArmTaskStatus;
using shopee_interfaces::msg::DetectedProduct;
using shopee_interfaces::msg::Pose6D;
using shopee_interfaces::srv::ArmMoveToPose;
using shopee_interfaces::srv::ArmPickProduct;
using shopee_interfaces::srv::ArmPlaceProduct;

class MockPackeeMain : public rclcpp::Node {
public:
  MockPackeeMain()
  : Node("mock_packee_main")
  {
    // 기본 파라미터
    this->declare_parameter<int>("robot_id", 1);
    this->declare_parameter<int>("order_id", 100);
    this->declare_parameter<std::string>("arm_side", "left");
    this->declare_parameter<int>("product_id", 501);

    robot_id_ = this->get_parameter("robot_id").as_int();
    order_id_ = this->get_parameter("order_id").as_int();
    product_id_ = this->get_parameter("product_id").as_int();
    arm_side_ = this->get_parameter("arm_side").as_string();

    RCLCPP_INFO(get_logger(), "✅ Mock Packee Main 초기화 완료 (robot_id=%d, order_id=%d, arm_side=%s)",
                robot_id_, order_id_, arm_side_.c_str());

    // 서비스 클라이언트 생성
    move_cli_ = this->create_client<ArmMoveToPose>("/packee/arm/move_to_pose");
    pick_cli_ = this->create_client<ArmPickProduct>("/packee/arm/pick_product");
    place_cli_ = this->create_client<ArmPlaceProduct>("/packee/arm/place_product");

    // 상태 구독
    pose_status_sub_ = this->create_subscription<ArmPoseStatus>(
      "/packee/arm/pose_status", 10,
      std::bind(&MockPackeeMain::OnPoseStatus, this, std::placeholders::_1));

    pick_status_sub_ = this->create_subscription<ArmTaskStatus>(
      "/packee/arm/pick_status", 10,
      std::bind(&MockPackeeMain::OnPickStatus, this, std::placeholders::_1));

    place_status_sub_ = this->create_subscription<ArmTaskStatus>(
      "/packee/arm/place_status", 10,
      std::bind(&MockPackeeMain::OnPlaceStatus, this, std::placeholders::_1));

    // 초기 테스트 시퀀스 실행 타이머
    timer_ = this->create_wall_timer(3s, std::bind(&MockPackeeMain::RunTestSequence, this));
  }

private:
  void RunTestSequence()
  {
    if (!move_cli_->wait_for_service(2s) ||
        !pick_cli_->wait_for_service(2s) ||
        !place_cli_->wait_for_service(2s)) {
      RCLCPP_WARN(get_logger(), "서비스 연결 대기 중...");
      return;
    }

    RCLCPP_INFO(get_logger(), "=== 🧩 PackeeArm 테스트 시작 ===");

    // 1️⃣ MoveToPose 요청
    auto move_req = std::make_shared<ArmMoveToPose::Request>();
    move_req->robot_id = robot_id_;
    move_req->order_id = order_id_;
    move_req->pose_type = "cart_view";

    auto move_future = move_cli_->async_send_request(move_req);
    RCLCPP_INFO(get_logger(), "➡️ MoveToPose 요청 보냄 (pose_type=cart_view)");

    // 2️⃣ PickProduct 요청
    auto pick_req = std::make_shared<ArmPickProduct::Request>();
    pick_req->robot_id = robot_id_;
    pick_req->order_id = order_id_;
    pick_req->arm_side = arm_side_;

    pick_req->target_product.product_id = product_id_;
    pick_req->target_product.confidence = 0.92F;
    pick_req->target_product.pose.x = 0.12;
    pick_req->target_product.pose.y = 0.02;
    pick_req->target_product.pose.z = 0.15;
    pick_req->target_product.pose.rz = 0.0;
    pick_req->target_product.bbox.x1 = 100;
    pick_req->target_product.bbox.y1 = 120;
    pick_req->target_product.bbox.x2 = 140;
    pick_req->target_product.bbox.y2 = 160;

    auto pick_future = pick_cli_->async_send_request(pick_req);
    RCLCPP_INFO(get_logger(), "➡️ PickProduct 요청 보냄 (product_id=%d, arm_side=%s)",
                product_id_, arm_side_.c_str());

    // 3️⃣ PlaceProduct 요청
    auto place_req = std::make_shared<ArmPlaceProduct::Request>();
    place_req->robot_id = robot_id_;
    place_req->order_id = order_id_;
    place_req->product_id = product_id_;
    place_req->arm_side = arm_side_;
    place_req->pose.x = 0.20;
    place_req->pose.y = -0.03;
    place_req->pose.z = 0.12;
    place_req->pose.rz = 0.0;

    auto place_future = place_cli_->async_send_request(place_req);
    RCLCPP_INFO(get_logger(), "➡️ PlaceProduct 요청 보냄 (product_id=%d, pose=(%.2f, %.2f, %.2f))",
                product_id_, place_req->pose.x, place_req->pose.y, place_req->pose.z);
  }

  // ------------------- 콜백 -------------------
  void OnPoseStatus(const ArmPoseStatus &msg)
  {
    RCLCPP_INFO(get_logger(),
                "📡 [PoseStatus] robot_id=%d order_id=%d pose_type=%s status=%s progress=%.2f",
                msg.robot_id, msg.order_id, msg.pose_type.c_str(),
                msg.status.c_str(), msg.progress);
  }

  void OnPickStatus(const ArmTaskStatus &msg)
  {
    RCLCPP_INFO(get_logger(),
                "🟢 [PickStatus] robot_id=%d order_id=%d product_id=%d arm_side=%s status=%s phase=%s progress=%.2f",
                msg.robot_id, msg.order_id, msg.product_id,
                msg.arm_side.c_str(), msg.status.c_str(),
                msg.current_phase.c_str(), msg.progress);
  }

  void OnPlaceStatus(const ArmTaskStatus &msg)
  {
    RCLCPP_INFO(get_logger(),
                "📦 [PlaceStatus] robot_id=%d order_id=%d product_id=%d arm_side=%s status=%s phase=%s progress=%.2f",
                msg.robot_id, msg.order_id, msg.product_id,
                msg.arm_side.c_str(), msg.status.c_str(),
                msg.current_phase.c_str(), msg.progress);
  }

  // ------------------- 멤버 -------------------
  int robot_id_{};
  int order_id_{};
  int product_id_{};
  std::string arm_side_;

  rclcpp::Client<ArmMoveToPose>::SharedPtr move_cli_;
  rclcpp::Client<ArmPickProduct>::SharedPtr pick_cli_;
  rclcpp::Client<ArmPlaceProduct>::SharedPtr place_cli_;

  rclcpp::Subscription<ArmPoseStatus>::SharedPtr pose_status_sub_;
  rclcpp::Subscription<ArmTaskStatus>::SharedPtr pick_status_sub_;
  rclcpp::Subscription<ArmTaskStatus>::SharedPtr place_status_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MockPackeeMain>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
