// packee_arm_controller.cpp
// 🟢 Pose6D 메시지를 joint_* / x,y,z 양쪽 포맷으로 대응하여 packee_main과의 연동을 담당한다.

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
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

// Pose6D가 x/y/z 필드를 사용하는지 식별한다.
template<typename PoseT, typename = void>
struct HasPoseXYZ : std::false_type {};

template<typename PoseT>
struct HasPoseXYZ<
  PoseT,
  std::void_t<
    decltype(std::declval<PoseT>().x),
    decltype(std::declval<PoseT>().y),
    decltype(std::declval<PoseT>().z),
    decltype(std::declval<PoseT>().rz)>> : std::true_type {};

// Pose6D가 joint_* 필드를 사용하는지 식별한다.
template<typename PoseT, typename = void>
struct HasPoseJoint : std::false_type {};

template<typename PoseT>
struct HasPoseJoint<
  PoseT,
  std::void_t<
    decltype(std::declval<PoseT>().joint_1),
    decltype(std::declval<PoseT>().joint_2),
    decltype(std::declval<PoseT>().joint_3),
    decltype(std::declval<PoseT>().joint_4)>> : std::true_type {};

// DetectedProduct.pose 필드 존재 여부를 통합적으로 확인한다.
template<typename T, typename = void>
struct HasPoseField : std::false_type {};

template<typename T>
struct HasPoseField<
  T,
  std::void_t<decltype(std::declval<T>().pose)>> : std::integral_constant<
    bool,
    HasPoseXYZ<decltype(std::declval<T>().pose)>::value ||
    HasPoseJoint<decltype(std::declval<T>().pose)>::value> {};

// packee_main(Mock)에서 position.x/y/z를 사용할 가능성도 대비한다.
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

namespace {
constexpr double kRadiansToDegrees = 57.29577951308232;  // rad → deg 변환 계수
}  // namespace

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
    pose_aliases_.emplace("ready_pose", "cart_view");

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
      this,
      max_translation_speed_,
      max_yaw_speed_deg_,
      command_timeout_sec_,
      left_velocity_topic_,
      right_velocity_topic_,
      velocity_frame_id_);

    gripper_ = std::make_unique<GripperController>(
      this,
      this->get_logger(),
      gripper_force_limit_,
      left_gripper_topic_,
      right_gripper_topic_);

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
  struct PoseComponents {
    double x;
    double y;
    double z;
    double yaw_deg;
  };

  // ---------------- 파라미터 선언 및 로드 ----------------
  void DeclareAndLoadParameters() {
    servo_gain_xy_ = this->declare_parameter("servo_gain_xy", 0.02);
    servo_gain_z_ = this->declare_parameter("servo_gain_z", 0.018);
    servo_gain_yaw_ = this->declare_parameter("servo_gain_yaw", 0.04);
    cnn_confidence_threshold_ = this->declare_parameter("cnn_confidence_threshold", 0.75);
    max_translation_speed_ = this->declare_parameter("max_translation_speed", 0.05);
    max_yaw_speed_deg_ = this->declare_parameter("max_yaw_speed_deg", 40.0);
    gripper_force_limit_ = this->declare_parameter("gripper_force_limit", 12.0);
    progress_publish_interval_sec_ = this->declare_parameter("progress_publish_interval", 0.15);
    command_timeout_sec_ = this->declare_parameter("command_timeout_sec", 4.0);
    left_velocity_topic_ = this->declare_parameter("left_arm_velocity_topic", "/packee/jetcobot/left/cmd_vel");
    right_velocity_topic_ = this->declare_parameter("right_arm_velocity_topic", "/packee/jetcobot/right/cmd_vel");
    velocity_frame_id_ = this->declare_parameter("velocity_frame_id", "packee_base");
    left_gripper_topic_ = this->declare_parameter("left_gripper_topic", "/packee/jetcobot/left/gripper_cmd");
    right_gripper_topic_ = this->declare_parameter("right_gripper_topic", "/packee/jetcobot/right/gripper_cmd");

    const std::vector<double> cart_view_values = this->declare_parameter<std::vector<double>>(
      "preset_pose_cart_view", {0.16, 0.0, 0.18, 0.0});
    const std::vector<double> standby_values = this->declare_parameter<std::vector<double>>(
      "preset_pose_standby", {0.10, 0.0, 0.14, 0.0});

    cart_view_preset_ = ParsePoseParameter(cart_view_values, "preset_pose_cart_view", default_cart_view_pose_);
    standby_preset_ = ParsePoseParameter(standby_values, "preset_pose_standby", default_standby_pose_);
  }

  // ---------------- Pose 파라미터 파싱 ----------------
  PoseEstimate ParsePoseParameter(const std::vector<double> &values,
                                  const std::string &name,
                                  const std::array<double, 4> &fallback) const {
    if (values.size() != 4U) {
      RCLCPP_WARN(this->get_logger(), "%s 파라미터는 4개의 값이 필요합니다.", name.c_str());
      return MakePoseFromArray(fallback);
    }
    PoseEstimate pose{};
    pose.x = values[0];
    pose.y = values[1];
    pose.z = values[2];
    pose.yaw_deg = values[3];
    pose.confidence = 1.0;
    return pose;
  }

  PoseEstimate MakePoseFromArray(const std::array<double, 4> &values) const {
    PoseEstimate p{};
    p.x = values[0];
    p.y = values[1];
    p.z = values[2];
    p.yaw_deg = values[3];
    p.confidence = 1.0;
    return p;
  }

  // ---------------- Pose 추출 헬퍼 ----------------
  template<typename PoseT>
  PoseComponents ConvertPoseGeneric(const PoseT &pose) const {
    PoseComponents c{};
    if constexpr (detail::HasPoseXYZ<PoseT>::value) {
      c.x = pose.x;
      c.y = pose.y;
      c.z = pose.z;
      c.yaw_deg = pose.rz * kRadiansToDegrees;
    } else if constexpr (detail::HasPoseJoint<PoseT>::value) {
      c.x = pose.joint_1;
      c.y = pose.joint_2;
      c.z = pose.joint_3;
      c.yaw_deg = pose.joint_4;
    }
    return c;
  }

  template<typename DetectedProductT>
  PoseComponents ExtractPoseFromDetectedProduct(const DetectedProductT &p) const {
    if constexpr (detail::HasPoseField<DetectedProductT>::value)
      return ConvertPoseGeneric(p.pose);
    else {
      PoseComponents c{};
      c.x = p.position.x;
      c.y = p.position.y;
      c.z = p.position.z;
      c.yaw_deg = 0.0;
      return c;
    }
  }

  // ---------------- 핸들러 ----------------
  // 이하 생략 (나머지 코드는 그대로 유지)
};
}  // namespace packee_arm

int main(int argc, char **argv) {
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
