// packee_arm_controller.cpp
// 🟢 [NOTE] Pose6D는 joint_1~joint_6 필드 기반 구조로 처리됩니다.

#include <array>
#include <cmath>
#include <memory>
#include <type_traits>
#include <utility>
#include <string>
#include <unordered_set>
#include <vector>

#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/arm_task_status.hpp"
#include "shopee_interfaces/msg/pose6_d.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/arm_place_product.hpp"

#include "packee_arm/arm_driver_proxy.hpp"
#include "packee_arm/constants.hpp"
#include "packee_arm/execution_manager.hpp"
#include "packee_arm/gripper_controller.hpp"
#include "packee_arm/types.hpp"
#include "packee_arm/visual_servo_module.hpp"

namespace packee_arm {

namespace detail {

// Pose6D 내부 필드 탐지용 템플릿
template<typename T, typename = void>
struct HasPoseField : std::false_type {};

template<typename T>
struct HasPoseField<
  T,
  std::void_t<
    decltype(std::declval<T>().pose.joint_1),
    decltype(std::declval<T>().pose.joint_2),
    decltype(std::declval<T>().pose.joint_3),
    decltype(std::declval<T>().pose.joint_4)>> : std::true_type {};

template<typename T, typename = void>
struct HasPositionField : std::false_type {};

template<typename T>
struct HasPositionField<
  T,
  std::void_t<
    decltype(std::declval<T>().position.x),
    decltype(std::declval<T>().position.y),
    decltype(std::declval<T>().position.z)>> : std::true_type {};

}  // namespace detail

using shopee_interfaces::msg::ArmPoseStatus;
using shopee_interfaces::msg::ArmTaskStatus;
using shopee_interfaces::srv::ArmMoveToPose;
using shopee_interfaces::srv::ArmPickProduct;
using shopee_interfaces::srv::ArmPlaceProduct;

// ------------------------------------------------------------------
// PackeeArmController 클래스 정의
// ------------------------------------------------------------------
class PackeeArmController : public rclcpp::Node {
public:
  PackeeArmController()
  : rclcpp::Node("packee_arm_controller"),
    valid_pose_types_({"cart_view", "standby"}),
    valid_arm_sides_({"left", "right"}) 
  {
    DeclareAndLoadParameters();

    pose_status_pub_ = this->create_publisher<ArmPoseStatus>("/packee/arm/pose_status", 10);
    pick_status_pub_ = this->create_publisher<ArmTaskStatus>("/packee/arm/pick_status", 10);
    place_status_pub_ = this->create_publisher<ArmTaskStatus>("/packee/arm/place_status", 10);

    move_service_ = this->create_service<ArmMoveToPose>(
      "/packee/arm/move_to_pose",
      std::bind(&PackeeArmController::HandleMoveToPose, this, std::placeholders::_1, std::placeholders::_2));
    pick_service_ = this->create_service<ArmPickProduct>(
      "/packee/arm/pick_product",
      std::bind(&PackeeArmController::HandlePickProduct, this, std::placeholders::_1, std::placeholders::_2));
    place_service_ = this->create_service<ArmPlaceProduct>(
      "/packee/arm/place_product",
      std::bind(&PackeeArmController::HandlePlaceProduct, this, std::placeholders::_1, std::placeholders::_2));

    visual_servo_ = std::make_unique<VisualServoModule>(
      servo_gain_xy_, servo_gain_z_, servo_gain_yaw_,
      cnn_confidence_threshold_, max_translation_speed_, max_yaw_speed_deg_);

    driver_ = std::make_unique<ArmDriverProxy>(
      this->get_logger(), max_translation_speed_, max_yaw_speed_deg_, command_timeout_sec_);

    gripper_ = std::make_unique<GripperController>(
      this->get_logger(), gripper_force_limit_);

    execution_manager_ = std::make_unique<ExecutionManager>(
      this,
      std::bind(&PackeeArmController::PublishPoseStatus, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6),
      std::bind(&PackeeArmController::PublishPickStatus, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
        std::placeholders::_7, std::placeholders::_8),
      std::bind(&PackeeArmController::PublishPlaceStatus, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
        std::placeholders::_7, std::placeholders::_8),
      visual_servo_.get(),
      driver_.get(),
      gripper_.get(),
      progress_publish_interval_sec_,
      command_timeout_sec_,
      cart_view_preset_,
      standby_preset_);

    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&PackeeArmController::OnParametersUpdated, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "✅ Packee Arm Controller 노드 초기화 완료");
  }

private:
  // ------------------------------------------------------------------
  // 내부 Pose 구조 정의
  struct PoseComponents {
    double x;
    double y;
    double z;
    double yaw_deg;
  };

  // ------------------------------------------------------------------
  // Pose6D 추출 (joint_1~joint_4 기준)
  PoseComponents ExtractPoseFromPoseMsg(const shopee_interfaces::msg::Pose6D &pose_msg) const {
    PoseComponents components{};
    components.x = static_cast<double>(pose_msg.joint_1);
    components.y = static_cast<double>(pose_msg.joint_2);
    components.z = static_cast<double>(pose_msg.joint_3);
    components.yaw_deg = static_cast<double>(pose_msg.joint_4);
    return components;
  }

  // ------------------------------------------------------------------
  // MoveToPose 핸들러
  void HandleMoveToPose(
    const std::shared_ptr<ArmMoveToPose::Request> request,
    std::shared_ptr<ArmMoveToPose::Response> response) 
  {
    if (!valid_pose_types_.count(request->pose_type)) {
      PublishPoseStatus(request->robot_id, request->order_id,
        request->pose_type, "failed", 0.0F, "지원되지 않는 pose_type입니다.");
      response->success = false;
      response->message = "유효하지 않은 pose_type입니다.";
      return;
    }

    MoveCommand cmd{};
    cmd.robot_id = request->robot_id;
    cmd.order_id = request->order_id;
    cmd.pose_type = request->pose_type;
    execution_manager_->EnqueueMove(cmd);

    response->success = true;
    response->message = "자세 변경 명령을 수락했습니다.";
  }

  // ------------------------------------------------------------------
  // PickProduct 핸들러
  void HandlePickProduct(
    const std::shared_ptr<ArmPickProduct::Request> request,
    std::shared_ptr<ArmPickProduct::Response> response)
  {
    if (!valid_arm_sides_.count(request->arm_side)) {
      PublishPickStatus(request->robot_id, request->order_id,
        request->target_product.product_id, request->arm_side,
        "failed", "servoing", 0.0F, "arm_side가 유효하지 않습니다.");
      response->success = false;
      response->message = "arm_side가 유효하지 않습니다.";
      return;
    }

    const PoseComponents pick_pose = ExtractPoseFromPoseMsg(request->target_product.pose);

    PickCommand cmd{};
    cmd.robot_id = request->robot_id;
    cmd.order_id = request->order_id;
    cmd.product_id = request->target_product.product_id;
    cmd.arm_side = request->arm_side;
    cmd.target_x = pick_pose.x;
    cmd.target_y = pick_pose.y;
    cmd.target_z = pick_pose.z;
    cmd.target_yaw_deg = pick_pose.yaw_deg;
    cmd.bbox_x1 = request->target_product.bbox.x1;
    cmd.bbox_y1 = request->target_product.bbox.y1;
    cmd.bbox_x2 = request->target_product.bbox.x2;
    cmd.bbox_y2 = request->target_product.bbox.y2;
    execution_manager_->EnqueuePick(cmd);

    response->success = true;
    response->message = "상품 픽업 명령을 수락했습니다.";
  }

  // ------------------------------------------------------------------
  // 🟢 [FIXED NOTE] PlaceProduct 핸들러 (Pose6D = joint_1~joint_4)
  void HandlePlaceProduct(
    const std::shared_ptr<ArmPlaceProduct::Request> request,
    std::shared_ptr<ArmPlaceProduct::Response> response)
  {
    if (!valid_arm_sides_.count(request->arm_side)) {
      PublishPlaceStatus(request->robot_id, request->order_id,
        request->product_id, request->arm_side,
        "failed", "servoing", 0.0F, "arm_side가 유효하지 않습니다.");
      response->success = false;
      response->message = "arm_side가 유효하지 않습니다.";
      return;
    }

    const PoseComponents place_pose = ExtractPoseFromPoseMsg(request->pose);

    if (!AreFinite(place_pose)) {
      PublishPlaceStatus(request->robot_id, request->order_id,
        request->product_id, request->arm_side,
        "failed", "servoing", 0.0F, "pose 값이 잘못되었습니다.");
      response->success = false;
      response->message = "pose 값이 잘못되었습니다.";
      return;
    }

    PlaceCommand cmd{};
    cmd.robot_id = request->robot_id;
    cmd.order_id = request->order_id;
    cmd.product_id = request->product_id;
    cmd.arm_side = request->arm_side;
    cmd.box_x = place_pose.x;
    cmd.box_y = place_pose.y;
    cmd.box_z = place_pose.z;
    cmd.box_yaw_deg = place_pose.yaw_deg;
    execution_manager_->EnqueuePlace(cmd);

    response->success = true;
    response->message = "상품 담기 명령을 수락했습니다.";
  }

  // ------------------------------------------------------------------
  // 상태 퍼블리셔 함수들
  void PublishPoseStatus(int32_t robot_id, int32_t order_id,
                         const std::string &pose_type,
                         const std::string &status, float progress,
                         const std::string &message)
  {
    ArmPoseStatus msg;
    msg.robot_id = robot_id;
    msg.order_id = order_id;
    msg.pose_type = pose_type;
    msg.status = status;
    msg.progress = progress;
    msg.message = message;
    pose_status_pub_->publish(msg);
  }

  void PublishPickStatus(int32_t robot_id, int32_t order_id, int32_t product_id,
                         const std::string &arm_side,
                         const std::string &status,
                         const std::string &phase,
                         float progress, const std::string &message)
  {
    ArmTaskStatus msg;
    msg.robot_id = robot_id;
    msg.order_id = order_id;
    msg.product_id = product_id;
    msg.arm_side = arm_side;
    msg.status = status;
    msg.current_phase = phase;
    msg.progress = progress;
    msg.message = message;
    pick_status_pub_->publish(msg);
  }

  void PublishPlaceStatus(int32_t robot_id, int32_t order_id, int32_t product_id,
                          const std::string &arm_side,
                          const std::string &status,
                          const std::string &phase,
                          float progress, const std::string &message)
  {
    ArmTaskStatus msg;
    msg.robot_id = robot_id;
    msg.order_id = order_id;
    msg.product_id = product_id;
    msg.arm_side = arm_side;
    msg.status = status;
    msg.current_phase = phase;
    msg.progress = progress;
    msg.message = message;
    place_status_pub_->publish(msg);
  }

  bool AreFinite(const PoseComponents &pose) const {
    return std::isfinite(pose.x) && std::isfinite(pose.y) &&
           std::isfinite(pose.z) && std::isfinite(pose.yaw_deg);
  }

  // ------------------------------------------------------------------
  // 멤버 변수
  rclcpp::Publisher<ArmPoseStatus>::SharedPtr pose_status_pub_;
  rclcpp::Publisher<ArmTaskStatus>::SharedPtr pick_status_pub_;
  rclcpp::Publisher<ArmTaskStatus>::SharedPtr place_status_pub_;

  rclcpp::Service<ArmMoveToPose>::SharedPtr move_service_;
  rclcpp::Service<ArmPickProduct>::SharedPtr pick_service_;
  rclcpp::Service<ArmPlaceProduct>::SharedPtr place_service_;

  std::unique_ptr<VisualServoModule> visual_servo_;
  std::unique_ptr<ArmDriverProxy> driver_;
  std::unique_ptr<GripperController> gripper_;
  std::unique_ptr<ExecutionManager> execution_manager_;

  std::unordered_set<std::string> valid_pose_types_;
  std::unordered_set<std::string> valid_arm_sides_;

  double servo_gain_xy_{0.02};
  double servo_gain_z_{0.018};
  double servo_gain_yaw_{0.04};
  double cnn_confidence_threshold_{0.75};
  double max_translation_speed_{0.05};
  double max_yaw_speed_deg_{40.0};
  double gripper_force_limit_{12.0};
  double progress_publish_interval_sec_{0.15};
  double command_timeout_sec_{4.0};

  PoseEstimate cart_view_preset_{};
  PoseEstimate standby_preset_{};
};

// ------------------------------------------------------------------
}  // namespace packee_arm

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<packee_arm::PackeeArmController>();
  try {
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node->get_logger(), "예외 발생: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
