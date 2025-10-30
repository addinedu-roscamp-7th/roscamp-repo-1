// packee_arm_controller.cpp
// 🟢 Pose6D 메시지를 joint_* / x,y,z 양쪽 포맷으로 대응하여 packee_main과의 연동을 담당한다.

// PackeeArmController 헤더를 포함한다.
#include "packee_arm/packee_arm_controller.hpp"

// 표준 알고리즘 유틸리티를 사용한다.
#include <algorithm>
// 스마트 포인터 사용을 위해 메모리 헤더를 포함한다.
#include <memory>

namespace packee_arm {

// PackeeArmController 생성자를 정의한다.
PackeeArmController::PackeeArmController()
// 노드 이름을 packee_arm_controller로 설정한다.
  : rclcpp::Node("packee_arm_controller"),
// 허용되는 pose 유형을 cart_view와 standby로 초기화한다.
    valid_pose_types_({"cart_view", "standby"}),
// 허용되는 팔 방향을 left와 right로 설정한다.
    valid_arm_sides_({"left", "right"})
{
    // 파라미터를 선언하고 초기값을 로드한다.
    DeclareAndLoadParameters();
    // packee_main이 사용하는 포즈 명칭을 미리 등록해 표준 pose_type으로 변환한다.
    pose_aliases_.emplace("ready_pose", "cart_view");

    // 포즈 상태를 발행하는 퍼블리셔를 생성한다.
    pose_status_pub_ = this->create_publisher<ArmPoseStatus>("/packee/arm/pose_status", 10);
    // 픽업 상태를 발행하는 퍼블리셔를 생성한다.
    pick_status_pub_ = this->create_publisher<ArmTaskStatus>("/packee/arm/pick_status", 10);
    // 포장 상태를 발행하는 퍼블리셔를 생성한다.
    place_status_pub_ = this->create_publisher<ArmTaskStatus>("/packee/arm/place_status", 10);

    // 자세 이동 요청을 처리하는 서비스를 등록한다.
    move_service_ = this->create_service<ArmMoveToPose>(
      "/packee/arm/move_to_pose",
      std::bind(&PackeeArmController::HandleMoveToPose, this, std::placeholders::_1, std::placeholders::_2));
    // 픽업 요청을 처리하는 서비스를 등록한다.
    pick_service_ = this->create_service<ArmPickProduct>(
      "/packee/arm/pick_product",
      std::bind(&PackeeArmController::HandlePickProduct, this, std::placeholders::_1, std::placeholders::_2));
    // 포장 요청을 처리하는 서비스를 등록한다.
    place_service_ = this->create_service<ArmPlaceProduct>(
      "/packee/arm/place_product",
      std::bind(&PackeeArmController::HandlePlaceProduct, this, std::placeholders::_1, std::placeholders::_2));

    // 시각 서보 모듈을 생성해 게인과 제한값을 전달한다.
    visual_servo_ = std::make_unique<VisualServoModule>(
      servo_gain_xy_, servo_gain_z_, servo_gain_yaw_,
      cnn_confidence_threshold_, max_translation_speed_, max_yaw_speed_deg_);

    // Arm 드라이버 프록시를 생성해 속도 제한과 토픽 정보를 지정한다.
    driver_ = std::make_unique<ArmDriverProxy>(
      this,
      max_translation_speed_,
      max_yaw_speed_deg_,
      command_timeout_sec_,
      left_velocity_topic_,
      right_velocity_topic_,
      velocity_frame_id_);
    // 그리퍼 컨트롤러를 생성해 힘 제한과 토픽을 설정한다.
    gripper_ = std::make_unique<GripperController>(
      this,
      this->get_logger(),
      gripper_force_limit_,
      left_gripper_topic_,
      right_gripper_topic_);
    // 실행 관리자를 생성해 하위 모듈과 콜백을 연결한다.
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

    // 파라미터 변경 콜백을 등록해 동적 갱신을 처리한다.
    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&PackeeArmController::OnParametersUpdated, this, std::placeholders::_1));

    // 노드 초기화 완료 로그를 출력한다.
    RCLCPP_INFO(this->get_logger(), "✅ Packee Arm Controller 노드 초기화 완료");
}

// 파라미터를 선언하고 내부 상태로 반영한다.
void PackeeArmController::DeclareAndLoadParameters()
{
    // 시각 서보 X/Y 게인을 선언한다.
    servo_gain_xy_ = this->declare_parameter<double>("servo_gain_xy", 0.02);
    // 시각 서보 Z 게인을 선언한다.
    servo_gain_z_ = this->declare_parameter<double>("servo_gain_z", 0.018);
    // 시각 서보 yaw 게인을 선언한다.
    servo_gain_yaw_ = this->declare_parameter<double>("servo_gain_yaw", 0.04);
    // CNN 신뢰도 임계값을 선언한다.
    cnn_confidence_threshold_ = this->declare_parameter<double>("cnn_confidence_threshold", 0.75);
    // 최대 병진 속도를 선언한다.
    max_translation_speed_ = this->declare_parameter<double>("max_translation_speed", 0.05);
    // 최대 yaw 속도를 선언한다.
    max_yaw_speed_deg_ = this->declare_parameter<double>("max_yaw_speed_deg", 40.0);
    // 그리퍼 힘 제한을 선언한다.
    gripper_force_limit_ = this->declare_parameter<double>("gripper_force_limit", 12.0);
    // 상태 발행 주기를 선언한다.
    progress_publish_interval_sec_ = this->declare_parameter<double>("progress_publish_interval", 0.15);
    // 명령 타임아웃을 선언한다.
    command_timeout_sec_ = this->declare_parameter<double>("command_timeout_sec", 4.0);
    // 좌측 팔 속도 토픽을 선언한다.
    left_velocity_topic_ = this->declare_parameter<std::string>("left_arm_velocity_topic", "/packee/jetcobot/left/cmd_vel");
    // 우측 팔 속도 토픽을 선언한다.
    right_velocity_topic_ = this->declare_parameter<std::string>("right_arm_velocity_topic", "/packee/jetcobot/right/cmd_vel");
    // 속도 메시지 프레임 아이디를 선언한다.
    velocity_frame_id_ = this->declare_parameter<std::string>("velocity_frame_id", "packee_base");
    // 좌측 그리퍼 토픽을 선언한다.
    left_gripper_topic_ = this->declare_parameter<std::string>("left_gripper_topic", "/packee/jetcobot/left/gripper_cmd");
    // 우측 그리퍼 토픽을 선언한다.
    right_gripper_topic_ = this->declare_parameter<std::string>("right_gripper_topic", "/packee/jetcobot/right/gripper_cmd");

    // cart_view 프리셋 포즈 파라미터를 선언한다.
    const std::vector<double> cart_view_values = this->declare_parameter<std::vector<double>>(
      "preset_pose_cart_view", {0.16, 0.0, 0.18, 0.0});
    // standby 프리셋 포즈 파라미터를 선언한다.
    const std::vector<double> standby_values = this->declare_parameter<std::vector<double>>(
      "preset_pose_standby", {0.10, 0.0, 0.14, 0.0});

    // cart_view 프리셋 값을 PoseEstimate로 변환해 저장한다.
    cart_view_preset_ = ParsePoseParameter(cart_view_values, "preset_pose_cart_view", default_cart_view_pose_);
    // standby 프리셋 값을 PoseEstimate로 변환해 저장한다.
    standby_preset_ = ParsePoseParameter(standby_values, "preset_pose_standby", default_standby_pose_);
}

PoseEstimate PackeeArmController::ParsePoseParameter(
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

PoseComponents PackeeArmController::ExtractPoseFromPoseMsg(const shopee_interfaces::msg::Pose6D & pose_msg) const
{
    return ConvertPoseGeneric(pose_msg);
}

PoseEstimate PackeeArmController::MakePoseFromArray(const std::array<double, 4> & values) const
{
    PoseEstimate pose{};
    pose.x = values[0];
    pose.y = values[1];
    pose.z = values[2];
    pose.yaw_deg = values[3];
    pose.confidence = 1.0;
    return pose;
}

template<typename PoseT>
PoseComponents PackeeArmController::ConvertPoseGeneric(const PoseT & pose) const
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

std::string PackeeArmController::NormalizePoseType(const std::string & pose_type) const
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

void PackeeArmController::HandleMoveToPose(
    const std::shared_ptr<ArmMoveToPose::Request> request,
    std::shared_ptr<ArmMoveToPose::Response> response)
{
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

void PackeeArmController::HandlePickProduct(
    const std::shared_ptr<ArmPickProduct::Request> request,
    std::shared_ptr<ArmPickProduct::Response> response)
{
    const int32_t product_id = request->product_id;
    if (product_id <= 0) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "product_id가 유효하지 않습니다.");
      response->success = false;
      response->message = "product_id가 유효하지 않습니다.";
      return;
    }

    if (!valid_arm_sides_.count(request->arm_side)) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "arm_side가 유효하지 않습니다.");
      response->success = false;
      response->message = "arm_side가 유효하지 않습니다.";
      return;
    }

    PoseComponents pick_pose = ExtractPoseFromPoseMsg(request->pose);
    if (!AreFinite(pick_pose) ||
        !std::isfinite(static_cast<double>(request->pose.rx)) ||
        !std::isfinite(static_cast<double>(request->pose.ry)) ||
        !std::isfinite(static_cast<double>(request->pose.rz))) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "pose에 유효하지 않은 값이 포함되어 있습니다.");
      response->success = false;
      response->message = "pose 값이 잘못되었습니다.";
      return;
    }

    if (!IsWithinWorkspace(pick_pose.x, pick_pose.y, pick_pose.z)) {
      ClampPoseToWorkspace(&pick_pose);
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "pose가 myCobot 280 작업 공간을 벗어났습니다.");
      response->success = false;
      response->message = "pose가 작업 공간 범위를 벗어났습니다.";
      return;
    }

    PickCommand command{};
    command.robot_id = request->robot_id;
    command.order_id = request->order_id;
    command.product_id = product_id;
    command.arm_side = request->arm_side;
    command.target_pose.x = pick_pose.x;
    command.target_pose.y = pick_pose.y;
    command.target_pose.z = pick_pose.z;
    command.target_pose.yaw_deg = pick_pose.yaw_deg;
    command.target_pose.confidence = 1.0;
    execution_manager_->EnqueuePick(command);
    response->success = true;
    response->message = "상품 픽업 명령을 수락했습니다.";
}

void PackeeArmController::HandlePlaceProduct(
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

    PoseComponents place_pose = ExtractPoseFromPoseMsg(request->pose);
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
    cmd.target_pose.x = place_pose.x;
    cmd.target_pose.y = place_pose.y;
    cmd.target_pose.z = place_pose.z;
    cmd.target_pose.yaw_deg = place_pose.yaw_deg;
    cmd.target_pose.confidence = 1.0;
    execution_manager_->EnqueuePlace(cmd);
    response->success = true;
    response->message = "상품 담기 명령을 수락했습니다.";
}

void PackeeArmController::PublishPoseStatus(int32_t robot_id, int32_t order_id,
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

void PackeeArmController::PublishPickStatus(int32_t robot_id, int32_t order_id, int32_t product_id,
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

void PackeeArmController::PublishPlaceStatus(int32_t robot_id, int32_t order_id, int32_t product_id,
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

rcl_interfaces::msg::SetParametersResult PackeeArmController::OnParametersUpdated(
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

bool PackeeArmController::AreFinite(double x, double y, double z) const
{
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

bool PackeeArmController::AreFinite(const PoseComponents & pose) const
{
    return AreFinite(pose.x, pose.y, pose.z) && std::isfinite(pose.yaw_deg);
}

bool PackeeArmController::IsZeroPose(const PoseComponents & pose) const
{
    constexpr double kTolerance = 1e-6;
    return std::fabs(pose.x) < kTolerance &&
           std::fabs(pose.y) < kTolerance &&
           std::fabs(pose.z) < kTolerance &&
           std::fabs(pose.yaw_deg) < kTolerance;
}

bool PackeeArmController::IsWithinWorkspace(double x, double y, double z) const
{
    const double radial = std::sqrt((x * x) + (y * y));
    if (radial > kMyCobotReach + 1e-6) {
      return false;
    }
    return z >= kMyCobotMinZ && z <= kMyCobotMaxZ;
}

void PackeeArmController::ClampPoseToWorkspace(PoseComponents * pose) const
{
    const double radial = std::sqrt((pose->x * pose->x) + (pose->y * pose->y));
    if (radial > kMyCobotReach) {
      const double scale = kMyCobotReach / std::max(radial, 1e-6);
      pose->x *= scale;
      pose->y *= scale;
    }
    pose->z = std::clamp(pose->z, kMyCobotMinZ, kMyCobotMaxZ);
}

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
