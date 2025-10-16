#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>

#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/packee_arm_task_status.hpp"
#include "shopee_interfaces/srv/packee_arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/packee_arm_pick_product.hpp"
#include "shopee_interfaces/srv/packee_arm_place_product.hpp"

using shopee_interfaces::msg::ArmPoseStatus;
using shopee_interfaces::msg::PackeeArmTaskStatus;
using shopee_interfaces::srv::PackeeArmMoveToPose;
using shopee_interfaces::srv::PackeeArmPickProduct;
using shopee_interfaces::srv::PackeeArmPlaceProduct;

class PackeeArmController : public rclcpp::Node {
public:
  PackeeArmController()
  : Node("packee_arm_controller"),
    valid_pose_types_({"cart_view", "standby"}),
    valid_arm_sides_({"left", "right"}) {
    // 자세 상태 퍼블리셔 생성
    pose_status_pub_ = this->create_publisher<ArmPoseStatus>("/packee/arm/pose_status", 10);
    // 픽업 상태 퍼블리셔 생성
    pick_status_pub_ = this->create_publisher<PackeeArmTaskStatus>("/packee/arm/pick_status", 10);
    // 담기 상태 퍼블리셔 생성
    place_status_pub_ = this->create_publisher<PackeeArmTaskStatus>("/packee/arm/place_status", 10);

    // 자세 변경 서비스 서버 생성
    move_service_ = this->create_service<PackeeArmMoveToPose>(
      "/packee/arm/move_to_pose",
      std::bind(&PackeeArmController::HandleMoveToPose, this, std::placeholders::_1, std::placeholders::_2));

    // 상품 픽업 서비스 서버 생성
    pick_service_ = this->create_service<PackeeArmPickProduct>(
      "/packee/arm/pick_product",
      std::bind(&PackeeArmController::HandlePickProduct, this, std::placeholders::_1, std::placeholders::_2));

    // 상품 담기 서비스 서버 생성
    place_service_ = this->create_service<PackeeArmPlaceProduct>(
      "/packee/arm/place_product",
      std::bind(&PackeeArmController::HandlePlaceProduct, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Packee Arm Controller 노드가 초기화되었습니다.");
  }

private:
  void HandleMoveToPose(
    const std::shared_ptr<PackeeArmMoveToPose::Request> request,
    std::shared_ptr<PackeeArmMoveToPose::Response> response) {
    // pose_type 유효성 검증
    if (valid_pose_types_.find(request->pose_type) == valid_pose_types_.end()) {
      PublishPoseStatus(request->robot_id, request->order_id, request->pose_type,
        "failed", 0.0F, "알 수 없는 포즈 타입입니다.");
      response->accepted = false;
      response->message = "지원되지 않는 포즈 타입입니다.";
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "자세 변경 요청 수신: robot_id=%d, order_id=%d, pose_type=%s",
      request->robot_id, request->order_id, request->pose_type.c_str());

    PublishPoseStatus(request->robot_id, request->order_id, request->pose_type,
      "in_progress", 0.5F, "자세 이동을 진행 중입니다.");

    PublishPoseStatus(request->robot_id, request->order_id, request->pose_type,
      "completed", 1.0F, "자세 이동을 완료했습니다.");

    response->accepted = true;
    response->message = "자세 변경 명령이 처리되었습니다.";
  }

  void HandlePickProduct(
    const std::shared_ptr<PackeeArmPickProduct::Request> request,
    std::shared_ptr<PackeeArmPickProduct::Response> response) {
    // arm_side 유효성 검증
    if (valid_arm_sides_.find(request->arm_side) == valid_arm_sides_.end()) {
      PublishPickStatus(request->robot_id, request->order_id, request->product_id, request->arm_side,
        "failed", "planning", 0.0F, "알 수 없는 팔 구분입니다.");
      response->accepted = false;
      response->message = "지원되지 않는 팔 구분입니다.";
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "상품 픽업 요청 수신: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s",
      request->robot_id, request->order_id, request->product_id, request->arm_side.c_str());

    PublishPickStatus(request->robot_id, request->order_id, request->product_id, request->arm_side,
      "in_progress", "planning", 0.3F, "픽업 경로를 계획 중입니다.");

    PublishPickStatus(request->robot_id, request->order_id, request->product_id, request->arm_side,
      "in_progress", "grasping", 0.6F, "상품을 파지하고 있습니다.");

    PublishPickStatus(request->robot_id, request->order_id, request->product_id, request->arm_side,
      "completed", "done", 1.0F, "상품 픽업을 완료했습니다.");

    response->accepted = true;
    response->message = "상품 픽업 명령이 처리되었습니다.";
  }

  void HandlePlaceProduct(
    const std::shared_ptr<PackeeArmPlaceProduct::Request> request,
    std::shared_ptr<PackeeArmPlaceProduct::Response> response) {
    // arm_side 유효성 검증
    if (valid_arm_sides_.find(request->arm_side) == valid_arm_sides_.end()) {
      PublishPlaceStatus(request->robot_id, request->order_id, request->product_id, request->arm_side,
        "failed", "planning", 0.0F, "알 수 없는 팔 구분입니다.");
      response->accepted = false;
      response->message = "지원되지 않는 팔 구분입니다.";
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "상품 담기 요청 수신: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s",
      request->robot_id, request->order_id, request->product_id, request->arm_side.c_str());

    PublishPlaceStatus(request->robot_id, request->order_id, request->product_id, request->arm_side,
      "in_progress", "approaching", 0.4F, "포장 박스로 이동 중입니다.");

    PublishPlaceStatus(request->robot_id, request->order_id, request->product_id, request->arm_side,
      "completed", "done", 1.0F, "상품 담기를 완료했습니다.");

    response->accepted = true;
    response->message = "상품 담기 명령이 처리되었습니다.";
  }

  void PublishPoseStatus(
    int32_t robot_id,
    int32_t order_id,
    const std::string & pose_type,
    const std::string & status,
    float progress,
    const std::string & message) {
    // 자세 상태 메시지 생성 및 발행
    auto status_msg = ArmPoseStatus();
    status_msg.robot_id = robot_id;
    status_msg.order_id = order_id;
    status_msg.pose_type = pose_type;
    status_msg.status = status;
    status_msg.progress = progress;
    status_msg.message = message;
    pose_status_pub_->publish(status_msg);
  }

  void PublishPickStatus(
    int32_t robot_id,
    int32_t order_id,
    int32_t product_id,
    const std::string & arm_side,
    const std::string & status,
    const std::string & current_phase,
    float progress,
    const std::string & message) {
    // 픽업 상태 메시지 생성 및 발행
    auto status_msg = PackeeArmTaskStatus();
    status_msg.robot_id = robot_id;
    status_msg.order_id = order_id;
    status_msg.product_id = product_id;
    status_msg.arm_side = arm_side;
    status_msg.status = status;
    status_msg.current_phase = current_phase;
    status_msg.progress = progress;
    status_msg.message = message;
    pick_status_pub_->publish(status_msg);
  }

  void PublishPlaceStatus(
    int32_t robot_id,
    int32_t order_id,
    int32_t product_id,
    const std::string & arm_side,
    const std::string & status,
    const std::string & current_phase,
    float progress,
    const std::string & message) {
    // 담기 상태 메시지 생성 및 발행
    auto status_msg = PackeeArmTaskStatus();
    status_msg.robot_id = robot_id;
    status_msg.order_id = order_id;
    status_msg.product_id = product_id;
    status_msg.arm_side = arm_side;
    status_msg.status = status;
    status_msg.current_phase = current_phase;
    status_msg.progress = progress;
    status_msg.message = message;
    place_status_pub_->publish(status_msg);
  }

  rclcpp::Publisher<ArmPoseStatus>::SharedPtr pose_status_pub_;
  rclcpp::Publisher<PackeeArmTaskStatus>::SharedPtr pick_status_pub_;
  rclcpp::Publisher<PackeeArmTaskStatus>::SharedPtr place_status_pub_;

  rclcpp::Service<PackeeArmMoveToPose>::SharedPtr move_service_;
  rclcpp::Service<PackeeArmPickProduct>::SharedPtr pick_service_;
  rclcpp::Service<PackeeArmPlaceProduct>::SharedPtr place_service_;

  const std::unordered_set<std::string> valid_pose_types_;
  const std::unordered_set<std::string> valid_arm_sides_;
};

int main(int argc, char ** argv) {
  // ROS 2 초기화
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PackeeArmController>();
  try {
    // 노드 스핀 수행
    rclcpp::spin(node);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "예외 발생: %s", error.what());
  }
  // 자원 정리
  rclcpp::shutdown();
  return 0;
}
