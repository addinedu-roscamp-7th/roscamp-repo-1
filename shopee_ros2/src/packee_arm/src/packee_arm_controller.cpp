// packee_arm_controller.cpp
// 🟢 Pose6D 메시지를 joint_* / x,y,z 양쪽 포맷으로 대응하여 packee_main과의 연동을 담당한다.

#include <algorithm>
// packee_arm_controller.cpp
// 🟢 Pose6D 메시지를 joint_* / x,y,z 양쪽 포맷으로 대응하여 packee_main과의 연동을 담당한다.

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
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
    valid_arm_sides_({"left", "right"}) 
  {
    DeclareAndLoadParameters();
    // packee_main이 사용하는 포즈 명칭을 미리 등록해 표준 pose_type으로 변환한다.
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

      servo_gain_xy_, servo_gain_z_, servo_gain_yaw_,
      cnn_confidence_threshold_, max_translation_speed_, max_yaw_speed_deg_);

    driver_ = std::make_unique<ArmDriverProxy>(
      this->get_logger(),
      max_translation_speed_,
      max_yaw_speed_deg_,
      command_timeout_sec_);
    gripper_ = std::make_unique<GripperController>(
      this->get_logger(),
      gripper_force_limit_);
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

  // 파라미터를 선언하고 기본값을 로드한다.
  // 노드 파라미터를 선언하고 기본값을 로드한다.
  void DeclareAndLoadParameters()
  {
    const double declared_servo_gain_xy =
      this->declare_parameter<double>("servo_gain_xy", 0.02);
    const double declared_servo_gain_z =
      this->declare_parameter<double>("servo_gain_z", 0.018);
    const double declared_servo_gain_yaw =
      this->declare_parameter<double>("servo_gain_yaw", 0.04);
    const double declared_confidence_threshold =
      this->declare_parameter<double>("cnn_confidence_threshold", 0.75);
    const double declared_max_translation_speed =
      this->declare_parameter<double>("max_translation_speed", 0.05);
    const double declared_max_yaw_speed_deg =
      this->declare_parameter<double>("max_yaw_speed_deg", 40.0);
    const double declared_gripper_force_limit =
      this->declare_parameter<double>("gripper_force_limit", 12.0);
    const double declared_progress_interval =
      this->declare_parameter<double>("progress_publish_interval", 0.15);
    const double declared_command_timeout =
      this->declare_parameter<double>("command_timeout_sec", 4.0);
    const std::string declared_left_velocity_topic =
      this->declare_parameter<std::string>("left_arm_velocity_topic", "/packee/jetcobot/left/cmd_vel");
    const std::string declared_right_velocity_topic =
      this->declare_parameter<std::string>("right_arm_velocity_topic", "/packee/jetcobot/right/cmd_vel");
    const std::string declared_velocity_frame =
      this->declare_parameter<std::string>("velocity_frame_id", "packee_base");
    // JetCobot 브릿지와 직접 연결되는 그리퍼 명령 토픽을 노출한다.
    const std::string declared_left_gripper_topic =
      this->declare_parameter<std::string>("left_gripper_topic", "/packee/jetcobot/left/gripper_cmd");
    const std::string declared_right_gripper_topic =
      this->declare_parameter<std::string>("right_gripper_topic", "/packee/jetcobot/right/gripper_cmd");

    const std::vector<double> cart_view_values = this->declare_parameter<std::vector<double>>(
      "preset_pose_cart_view", {0.16, 0.0, 0.18, 0.0});
    const std::vector<double> standby_values = this->declare_parameter<std::vector<double>>(
      "preset_pose_standby", {0.10, 0.0, 0.14, 0.0});

    cart_view_preset_ = ParsePoseParameter(cart_view_values, "preset_pose_cart_view", default_cart_view_pose_);
    standby_preset_ = ParsePoseParameter(standby_values, "preset_pose_standby", default_standby_pose_);
  }

  // 파라미터 벡터를 PoseEstimate로 변환하며 유효성 검사를 수행한다.
  // 파라미터 배열을 PoseEstimate로 변환하며 값 검증을 수행한다.
  PoseEstimate ParsePoseParameter(
    const std::vector<double> & values,
    const std::string & parameter_name,
    const std::array<double, 4> & fallback) const
  {
    if (values.size() != 4U) {
      RCLCPP_WARN(
        this->get_logger(),
        "%s 파라미터는 4개의 값을 가져야 합니다. 기본값을 사용합니다.",
        parameter_name.c_str());
      return MakePoseFromArray(fallback);
    }
    const double x = values[0];
    const double y = values[1];
    const double z = values[2];
    const double yaw_deg = values[3];
    if (!AreFinite(x, y, z) || !std::isfinite(yaw_deg)) {
      RCLCPP_WARN(
        this->get_logger(),
        "%s 파라미터에 유한하지 않은 값이 포함되어 기본값을 사용합니다.",
        parameter_name.c_str());
      return MakePoseFromArray(fallback);
    }
    if (!IsWithinWorkspace(x, y, z)) {
      RCLCPP_WARN(
        this->get_logger(),
        "%s 파라미터가 myCobot 280 작업 공간을 벗어납니다. 기본값을 사용합니다.",
        parameter_name.c_str());
      return MakePoseFromArray(fallback);
    }
    PoseEstimate pose{};
    pose.x = x;
    pose.y = y;
    pose.z = z;
    pose.yaw_deg = yaw_deg;
    pose.confidence = 1.0;
    return pose;
  }

  PoseComponents ExtractPoseFromPoseMsg(const shopee_interfaces::msg::Pose6D & pose_msg) const {
    // Pose6D 메시지를 x, y, z, yaw_deg로 변환한다.
    PoseComponents components{};
    components.x = static_cast<double>(pose_msg.x);
    components.y = static_cast<double>(pose_msg.y);
    components.z = static_cast<double>(pose_msg.z);
    components.yaw_deg = static_cast<double>(pose_msg.rz) * kRadiansToDegrees;
    return components;
  }

  template<typename DetectedProductT>
  PoseComponents ExtractPoseFromDetectedProduct(const DetectedProductT & product) const {
    static_assert(detail::HasPoseField<DetectedProductT>::value, "DetectedProduct에 pose 필드가 필요합니다.");
    return ExtractPoseFromPoseMsg(product.pose);
  }

  PoseEstimate MakePoseFromArray(const std::array<double, 4> & values) const {
    // 배열 형태의 기본값을 PoseEstimate로 치환한다.
    PoseEstimate pose{};
    pose.x = values[0];
    pose.y = values[1];
    pose.z = values[2];
    pose.yaw_deg = values[3];
    pose.confidence = 1.0;
    return pose;
  }

  // Pose6D 추출 (joint_* 포맷과 x/y/z 포맷을 모두 수용)
  PoseComponents ExtractPoseFromPoseMsg(const shopee_interfaces::msg::Pose6D & pose_msg) const
  {
    return ConvertPoseGeneric(pose_msg);
  }

  // DetectedProduct에서 Pose 또는 position 필드를 이용해 포즈를 복원한다.
  template<typename DetectedProductT>
  PoseComponents ExtractPoseFromDetectedProduct(const DetectedProductT & product) const
  {
    static_assert(
      detail::HasPoseField<DetectedProductT>::value || detail::HasPositionField<DetectedProductT>::value,
      "DetectedProduct에 pose 또는 position 정보가 필요합니다.");

    if constexpr (detail::HasPoseField<DetectedProductT>::value) {
      return ConvertPoseGeneric(product.pose);
    } else {
      PoseComponents components{};
      components.x = static_cast<double>(product.position.x);
      components.y = static_cast<double>(product.position.y);
      components.z = static_cast<double>(product.position.z);
      components.yaw_deg = 0.0;  // position만 제공되면 yaw는 0으로 처리
      return components;
    }
  }

  // Pose6D 또는 이에 준하는 구조체를 공통 포맷(PoseComponents)으로 변환한다.
  template<typename PoseT>
  PoseComponents ConvertPoseGeneric(const PoseT & pose) const
  {
    PoseComponents components{};
    if constexpr (detail::HasPoseXYZ<PoseT>::value) {
      // x, y, z, rx, ry, rz 필드를 직접 활용하는 포맷
      components.x = static_cast<double>(pose.x);
      components.y = static_cast<double>(pose.y);
      components.z = static_cast<double>(pose.z);
      components.yaw_deg = static_cast<double>(pose.rz);
    } else if constexpr (detail::HasPoseJoint<PoseT>::value) {
      // 기존 joint_* 포맷에 대한 하위 호환 처리
      components.x = static_cast<double>(pose.joint_1);
      components.y = static_cast<double>(pose.joint_2);
      components.z = static_cast<double>(pose.joint_3);
      components.yaw_deg = static_cast<double>(pose.joint_4);
    } else {
      static_assert(detail::HasPoseXYZ<PoseT>::value || detail::HasPoseJoint<PoseT>::value,
        "Pose6D 구조에 지원되지 않는 필드 세트입니다.");
    }
    return components;
  }

  // 외부에서 전달된 pose_type을 내부에서 사용하는 명칭으로 정규화한다.
  std::string NormalizePoseType(const std::string & pose_type) const
  {
    if (valid_pose_types_.count(pose_type) > 0) {
      return pose_type;
    }
    const auto alias_iter = pose_aliases_.find(pose_type);
    if (alias_iter != pose_aliases_.end()) {
      return alias_iter->second;
    }
    return {};
  }

  // ------------------------------------------------------------------
  // MoveToPose 핸들러
  void HandleMoveToPose(
    const std::shared_ptr<ArmMoveToPose::Request> request,
    std::shared_ptr<ArmMoveToPose::Response> response)
  {
    // packee_main에서 사용하는 별칭(ready_pose 등)을 표준 pose_type으로 변환한다.
    const std::string normalized_pose = NormalizePoseType(request->pose_type);
    if (normalized_pose.empty()) {
      PublishPoseStatus(
        request->robot_id,
        request->order_id,
        request->pose_type,
        "failed",
        0.0F,
        "지원되지 않는 pose_type입니다.");
      response->success = false;
      response->message = "유효하지 않은 pose_type입니다.";
      return;
    }

    MoveCommand cmd{};
    cmd.robot_id = request->robot_id;
    cmd.order_id = request->order_id;
    cmd.pose_type = normalized_pose;
    execution_manager_->EnqueueMove(cmd);

    response->success = true;
    response->message = "자세 변경 명령을 수락했습니다.";
  }

  // ------------------------------------------------------------------
  // PickProduct 핸들러
  // Packee Main이 전달한 상품 픽업 요청을 처리한다.
  void HandlePickProduct(
    const std::shared_ptr<ArmPickProduct::Request> request,
    std::shared_ptr<ArmPickProduct::Response> response)
  {
    if (!valid_arm_sides_.count(request->arm_side)) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->target_product.product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "arm_side가 유효하지 않습니다.");
      response->success = false;
      response->message = "arm_side가 유효하지 않습니다.";
      return;
    }

    if (!IsValidBoundingBox(
        request->target_product.bbox.x1,
        request->target_product.bbox.y1,
        request->target_product.bbox.x2,
        request->target_product.bbox.y2)) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->target_product.product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "bbox 좌표가 유효하지 않습니다.");
      response->success = false;
      response->message = "bbox가 유효하지 않습니다.";
      return;
    }

    const double detection_confidence =
      static_cast<double>(request->target_product.confidence);
    if (!std::isfinite(detection_confidence) || detection_confidence <= 0.0 ||
        detection_confidence > 1.0) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->target_product.product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "target_product.confidence가 유효 범위를 벗어났습니다.");
      response->success = false;
      response->message = "confidence 값이 잘못되었습니다.";
      return;
    }

    if (detection_confidence < cnn_confidence_threshold_) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->target_product.product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "target_product.confidence가 cnn_confidence_threshold보다 낮습니다.");
      response->success = false;
      response->message = "confidence가 임계값보다 낮습니다.";
      return;
    }

    const PoseComponents pick_pose = ExtractPoseFromDetectedProduct(request->target_product);

    if (!AreFinite(pick_pose) ||
        !std::isfinite(static_cast<double>(request->target_product.pose.rx)) ||
        !std::isfinite(static_cast<double>(request->target_product.pose.ry)) ||
        !std::isfinite(static_cast<double>(request->target_product.pose.rz))) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->target_product.product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "target_product.pose에 유효하지 않은 값이 포함되어 있습니다.");
      response->success = false;
      response->message = "pose 값이 잘못되었습니다.";
      return;
    }

    if (!IsWithinWorkspace(pick_pose.x, pick_pose.y, pick_pose.z)) {
      // packee_main이 범위를 벗어난 좌표를 보낼 수 있으므로 안전 범위로 보정한다.
      ClampPoseToWorkspace(&pick_pose);
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->target_product.product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "target_product.pose가 myCobot 280 작업 공간을 벗어났습니다.");
      response->success = false;
      response->message = "pose가 작업 공간 범위를 벗어났습니다.";
      return;
    }

    PickCommand command{};
    command.robot_id = request->robot_id;
    command.order_id = request->order_id;
    command.product_id = request->target_product.product_id;
    command.arm_side = request->arm_side;
    command.target_pose.x = pick_pose.x;
    command.target_pose.y = pick_pose.y;
    command.target_pose.z = pick_pose.z;
    command.target_pose.yaw_deg = pick_pose.yaw_deg;
    command.target_pose.confidence = detection_confidence;
    command.detection_confidence = detection_confidence;
    command.bbox_x1 = request->target_product.bbox.x1;
    command.bbox_y1 = request->target_product.bbox.y1;
    command.bbox_x2 = request->target_product.bbox.x2;
    command.bbox_y2 = request->target_product.bbox.y2;
    execution_manager_->EnqueuePick(command);
    response->success = true;
    response->message = "상품 픽업 명령을 수락했습니다.";
  }

  // Packee Main이 전달한 상품 담기 요청을 처리한다.
  void HandlePlaceProduct(
    const std::shared_ptr<ArmPlaceProduct::Request> request,
    std::shared_ptr<ArmPlaceProduct::Response> response)
  {
    if (!valid_arm_sides_.count(request->arm_side)) {
      PublishPlaceStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "arm_side가 유효하지 않습니다.");
      response->success = false;
      response->message = "arm_side가 유효하지 않습니다.";
      return;
    }

    const PoseComponents place_pose = ExtractPoseFromPoseMsg(request->pose);
    if (!AreFinite(place_pose) ||
        !std::isfinite(static_cast<double>(request->pose.rx)) ||
        !std::isfinite(static_cast<double>(request->pose.ry)) ||
        !std::isfinite(static_cast<double>(request->pose.rz))) {
      PublishPlaceStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "pose 값이 유효하지 않습니다.");
      response->success = false;
      response->message = "pose 값이 유효하지 않습니다.";
      return;
    }

    if (IsZeroPose(place_pose)) {
      // 좌표가 제공되지 않은 경우 standby 프리셋 사용
      place_pose.x = standby_preset_.x;
      place_pose.y = standby_preset_.y;
      place_pose.z = standby_preset_.z;
      place_pose.yaw_deg = standby_preset_.yaw_deg;
      PublishPlaceStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "in_progress",
        "planning",
        0.05F,
        "pose가 제공되지 않아 standby 프리셋을 적용했습니다.");
    }

    if (!IsWithinWorkspace(place_pose.x, place_pose.y, place_pose.z)) {
      // 담기 위치도 안전 범위로 클램프한다.
      ClampPoseToWorkspace(&place_pose);
      PublishPlaceStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "pose가 myCobot 280 작업 공간을 벗어났습니다.");
      response->success = false;
      response->message = "pose가 작업 공간 범위를 벗어났습니다.";
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

  // 파라미터 동적 업데이트를 처리하여 런타임 파라미터 조정을 허용한다.
  rcl_interfaces::msg::SetParametersResult OnParametersUpdated(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "파라미터가 업데이트되었습니다.";

    double next_servo_gain_xy = servo_gain_xy_;
    double next_servo_gain_z = servo_gain_z_;
    double next_servo_gain_yaw = servo_gain_yaw_;
    double next_confidence_threshold = cnn_confidence_threshold_;
    double next_max_translation_speed = max_translation_speed_;
    double next_max_yaw_speed_deg = max_yaw_speed_deg_;
    double next_gripper_force_limit = gripper_force_limit_;
    double next_progress_interval = progress_publish_interval_sec_;
    double next_command_timeout = command_timeout_sec_;
    PoseEstimate next_cart_view_preset = cart_view_preset_;
    PoseEstimate next_standby_preset = standby_preset_;

    for (const auto & parameter : parameters) {
      const std::string & name = parameter.get_name();
      if (name == "servo_gain_xy") {
        next_servo_gain_xy = parameter.as_double();
      } else if (name == "servo_gain_z") {
        next_servo_gain_z = parameter.as_double();
      } else if (name == "servo_gain_yaw") {
        next_servo_gain_yaw = parameter.as_double();
      } else if (name == "cnn_confidence_threshold") {
        next_confidence_threshold = parameter.as_double();
      } else if (name == "max_translation_speed") {
        next_max_translation_speed = parameter.as_double();
      } else if (name == "max_yaw_speed_deg") {
        next_max_yaw_speed_deg = parameter.as_double();
      } else if (name == "gripper_force_limit") {
        next_gripper_force_limit = parameter.as_double();
      } else if (name == "progress_publish_interval") {
        next_progress_interval = parameter.as_double();
      } else if (name == "command_timeout_sec") {
        next_command_timeout = parameter.as_double();
      } else if (name == "preset_pose_cart_view") {
        next_cart_view_preset = ParsePoseParameter(
          parameter.as_double_array(),
          "preset_pose_cart_view",
          default_cart_view_pose_);
      } else if (name == "preset_pose_standby") {
        next_standby_preset = ParsePoseParameter(
          parameter.as_double_array(),
          "preset_pose_standby",
          default_standby_pose_);
      }
    }

    if (next_servo_gain_xy <= 0.0 || next_servo_gain_z <= 0.0 || next_servo_gain_yaw <= 0.0) {
      result.successful = false;
      result.reason = "servo 게인은 0보다 커야 합니다.";
      return result;
    }
    if (next_confidence_threshold <= 0.0 || next_confidence_threshold > 1.0) {
      result.successful = false;
      result.reason = "cnn_confidence_threshold는 (0, 1] 범위여야 합니다.";
      return result;
    }
    if (next_max_translation_speed <= 0.0 || next_max_yaw_speed_deg <= 0.0) {
      result.successful = false;
      result.reason = "최대 속도 파라미터는 0보다 커야 합니다.";
      return result;
    }
    if (next_gripper_force_limit <= 0.0) {
      result.successful = false;
      result.reason = "gripper_force_limit은 0보다 커야 합니다.";
      return result;
    }
    if (next_progress_interval <= 0.0) {
      result.successful = false;
      result.reason = "progress_publish_interval은 0보다 커야 합니다.";
      return result;
    }
    if (next_command_timeout <= 0.0) {
      result.successful = false;
      result.reason = "command_timeout_sec은 0보다 커야 합니다.";
      return result;
    }

    servo_gain_xy_ = next_servo_gain_xy;
    servo_gain_z_ = next_servo_gain_z;
    servo_gain_yaw_ = next_servo_gain_yaw;
    cnn_confidence_threshold_ = next_confidence_threshold;
    max_translation_speed_ = next_max_translation_speed;
    max_yaw_speed_deg_ = next_max_yaw_speed_deg;
    gripper_force_limit_ = next_gripper_force_limit;
    progress_publish_interval_sec_ = next_progress_interval;
    command_timeout_sec_ = next_command_timeout;
    cart_view_preset_ = next_cart_view_preset;
    standby_preset_ = next_standby_preset;

    visual_servo_->UpdateGains(servo_gain_xy_, servo_gain_z_, servo_gain_yaw_);
    visual_servo_->UpdateConstraints(
      cnn_confidence_threshold_,
      max_translation_speed_,
      max_yaw_speed_deg_);
    gripper_->UpdateForceLimit(gripper_force_limit_);
    driver_->UpdateConstraints(
      max_translation_speed_,
      max_yaw_speed_deg_,
      command_timeout_sec_);
    execution_manager_->UpdateTiming(
      progress_publish_interval_sec_,
      command_timeout_sec_);
    execution_manager_->UpdatePosePresets(
      cart_view_preset_,
      standby_preset_);

    return result;
  }

  bool AreFinite(double x, double y, double z) const
  {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  bool AreFinite(const PoseComponents & pose) const
  {
    return AreFinite(pose.x, pose.y, pose.z) && std::isfinite(pose.yaw_deg);
  }

  bool IsZeroPose(const PoseComponents & pose) const
  {
    constexpr double kTolerance = 1e-6;
    return std::fabs(pose.x) < kTolerance &&
           std::fabs(pose.y) < kTolerance &&
           std::fabs(pose.z) < kTolerance &&
           std::fabs(pose.yaw_deg) < kTolerance;
  }

  bool IsWithinWorkspace(double x, double y, double z) const
  {
    const double radial = std::sqrt((x * x) + (y * y));
    if (radial > kMyCobotReach + 1e-6) {
      return false;
    }
    return z >= kMyCobotMinZ && z <= kMyCobotMaxZ;
  }

  void ClampPoseToWorkspace(PoseComponents * pose) const
  {
    const double radial = std::sqrt((pose->x * pose->x) + (pose->y * pose->y));
    if (radial > kMyCobotReach) {
      const double scale = kMyCobotReach / std::max(radial, 1e-6);
      pose->x *= scale;
      pose->y *= scale;
    }
    pose->z = std::clamp(pose->z, kMyCobotMinZ, kMyCobotMaxZ);
  }

  bool IsValidBoundingBox(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const
  {
    return x2 > x1 && y2 > y1;
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
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  std::unordered_set<std::string> valid_pose_types_;
  std::unordered_set<std::string> valid_arm_sides_;
  std::unordered_map<std::string, std::string> pose_aliases_;

  double servo_gain_xy_{0.02};
  double servo_gain_z_{0.018};
  double servo_gain_yaw_{0.04};
  double cnn_confidence_threshold_{0.75};
  double max_translation_speed_{0.05};
  double max_yaw_speed_deg_{30.0};
  double gripper_force_limit_{12.0};
  double progress_publish_interval_sec_{0.15};
  double command_timeout_sec_{4.0};
  std::string left_velocity_topic_{"/packee/jetcobot/left/cmd_vel"};
  std::string right_velocity_topic_{"/packee/jetcobot/right/cmd_vel"};
  std::string velocity_frame_id_{"packee_base"};
  std::string left_gripper_topic_{"/packee/jetcobot/left/gripper_cmd"};
  std::string right_gripper_topic_{"/packee/jetcobot/right/gripper_cmd"};
  const std::array<double, 4> default_cart_view_pose_{{0.16, 0.0, 0.18, 0.0}};  // 카트 확인 자세
  const std::array<double, 4> default_standby_pose_{{0.0, 0.0, 0.0, 0.0}};  // 대기 자세
  PoseEstimate cart_view_preset_{};
  PoseEstimate standby_preset_{};
  const std::array<double, 4> default_cart_view_pose_{{0.16, 0.0, 0.18, 0.0}};
  const std::array<double, 4> default_standby_pose_{{0.10, 0.0, 0.14, 0.0}};
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
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node->get_logger(), "예외 발생: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
