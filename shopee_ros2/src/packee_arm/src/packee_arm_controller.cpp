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
    decltype(std::declval<T>().pose.x),
    decltype(std::declval<T>().pose.y),
    decltype(std::declval<T>().pose.z),
    decltype(std::declval<T>().pose.rz)>> : std::true_type {};

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
    // JetCobot 브릿지가 float32 명령을 받아 실제 그리퍼를 여닫도록 퍼블리셔를 구성한다.
    gripper_ = std::make_unique<GripperController>(
      this,
      this->get_logger(),
      gripper_force_limit_,
      left_gripper_topic_,
      right_gripper_topic_);

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

  void DeclareAndLoadParameters() {
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
      "preset_pose_cart_view",
      std::vector<double>(default_cart_view_pose_.begin(), default_cart_view_pose_.end()));
    const std::vector<double> standby_values = this->declare_parameter<std::vector<double>>(
      "preset_pose_standby",
      std::vector<double>(default_standby_pose_.begin(), default_standby_pose_.end()));

    servo_gain_xy_ = declared_servo_gain_xy;
    servo_gain_z_ = declared_servo_gain_z;
    servo_gain_yaw_ = declared_servo_gain_yaw;
    cnn_confidence_threshold_ = declared_confidence_threshold;
    max_translation_speed_ = declared_max_translation_speed;
    max_yaw_speed_deg_ = declared_max_yaw_speed_deg;
    gripper_force_limit_ = declared_gripper_force_limit;
    progress_publish_interval_sec_ = declared_progress_interval;
    command_timeout_sec_ = declared_command_timeout;
    left_velocity_topic_ = declared_left_velocity_topic;
    right_velocity_topic_ = declared_right_velocity_topic;
    velocity_frame_id_ = declared_velocity_frame;
    left_gripper_topic_ = declared_left_gripper_topic;
    right_gripper_topic_ = declared_right_gripper_topic;

    cart_view_preset_ = ParsePoseParameter(
      cart_view_values,
      "preset_pose_cart_view",
      default_cart_view_pose_);
    standby_preset_ = ParsePoseParameter(
      standby_values,
      "preset_pose_standby",
      default_standby_pose_);
  }

  PoseEstimate ParsePoseParameter(
    const std::vector<double> & values,
    const std::string & parameter_name,
    const std::array<double, 4> & fallback) const {
    // 파라미터 배열을 PoseEstimate로 변환하며 범위 검증을 수행한다.
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
    const double yaw = values[3];
    if (!AreFinite(x, y, z) || !std::isfinite(yaw)) {
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
    pose.yaw_deg = yaw;
    pose.confidence = 1.0;
    return pose;
  }

  PoseComponents ExtractPoseFromPoseMsg(const shopee_interfaces::msg::Pose6D & pose_msg) const {
    // Pose6D 메시지를 x, y, z, yaw_deg로 변환한다.

  // ------------------------------------------------------------------
  // Pose6D 추출 (joint_1~joint_4 기준)
  PoseComponents ExtractPoseFromPoseMsg(const shopee_interfaces::msg::Pose6D &pose_msg) const {

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

  bool IsWithinWorkspace(double x, double y, double z) const {
    // myCobot 280 작업 공간(수평 반경, 높이)을 벗어나는지 확인한다.
    const double radial = std::sqrt((x * x) + (y * y));
    if (radial > kMyCobotReach + 1e-6) {
      return false;
    }
    if (z < kMyCobotMinZ || z > kMyCobotMaxZ) {
      return false;
    }
    return true;
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

      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->target_product.product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "arm_side가 유효하지 않습니다.");

      PublishPickStatus(request->robot_id, request->order_id,
        request->target_product.product_id, request->arm_side,
        "failed", "servoing", 0.0F, "arm_side가 유효하지 않습니다."); 
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

      PublishPlaceStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "arm_side가 유효하지 않습니다.");

      PublishPlaceStatus(request->robot_id, request->order_id,
        request->product_id, request->arm_side,
        "failed", "servoing", 0.0F, "arm_side가 유효하지 않습니다.");

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
        "pose에 유효하지 않은 값이 포함되어 있습니다.");


    if (!AreFinite(place_pose)) {
      PublishPlaceStatus(request->robot_id, request->order_id,
        request->product_id, request->arm_side,
        "failed", "servoing", 0.0F, "pose 값이 잘못되었습니다.");

      response->success = false;
      response->message = "pose 값이 잘못되었습니다.";
      return;
    }


    if (!IsWithinWorkspace(place_pose.x, place_pose.y, place_pose.z)) {
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

    PlaceCommand command{};
    command.robot_id = request->robot_id;
    command.order_id = request->order_id;
    command.product_id = request->product_id;
    command.arm_side = request->arm_side;
    command.target_pose.x = place_pose.x;
    command.target_pose.y = place_pose.y;
    command.target_pose.z = place_pose.z;
    command.target_pose.yaw_deg = place_pose.yaw_deg;
    command.target_pose.confidence = 1.0;
    execution_manager_->EnqueuePlace(command);

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
