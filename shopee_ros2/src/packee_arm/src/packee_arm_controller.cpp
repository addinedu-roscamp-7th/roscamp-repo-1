#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/packee_arm_task_status.hpp"
#include "shopee_interfaces/srv/packee_arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/packee_arm_pick_product.hpp"
#include "shopee_interfaces/srv/packee_arm_place_product.hpp"

#include "packee_arm/arm_driver_proxy.hpp"
#include "packee_arm/constants.hpp"
#include "packee_arm/execution_manager.hpp"
#include "packee_arm/gripper_controller.hpp"
#include "packee_arm/types.hpp"
#include "packee_arm/visual_servo_module.hpp"

namespace packee_arm {

using shopee_interfaces::msg::ArmPoseStatus;
using shopee_interfaces::msg::PackeeArmTaskStatus;
using shopee_interfaces::srv::PackeeArmMoveToPose;
using shopee_interfaces::srv::PackeeArmPickProduct;
using shopee_interfaces::srv::PackeeArmPlaceProduct;

// PackeeArmController 클래스는 ROS 인터페이스, 파라미터 관리, 상태 발행을 담당한다.
class PackeeArmController : public rclcpp::Node {
public:
  PackeeArmController()
  : rclcpp::Node("packee_arm_controller"),
    valid_pose_types_({"cart_view", "standby"}),
    valid_arm_sides_({"left", "right"}) {
    // myCobot 280을 위한 기본 파라미터와 자세 프리셋을 로드한다.
    DeclareAndLoadParameters();

    pose_status_pub_ = this->create_publisher<ArmPoseStatus>(
      "/packee/arm/pose_status",
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    pick_status_pub_ = this->create_publisher<PackeeArmTaskStatus>(
      "/packee/arm/pick_status",
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    place_status_pub_ = this->create_publisher<PackeeArmTaskStatus>(
      "/packee/arm/place_status",
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable());

    move_service_ = this->create_service<PackeeArmMoveToPose>(
      "/packee/arm/move_to_pose",
      std::bind(&PackeeArmController::HandleMoveToPose, this, std::placeholders::_1, std::placeholders::_2));
    pick_service_ = this->create_service<PackeeArmPickProduct>(
      "/packee/arm/pick_product",
      std::bind(&PackeeArmController::HandlePickProduct, this, std::placeholders::_1, std::placeholders::_2));
    place_service_ = this->create_service<PackeeArmPlaceProduct>(
      "/packee/arm/place_product",
      std::bind(&PackeeArmController::HandlePlaceProduct, this, std::placeholders::_1, std::placeholders::_2));

    visual_servo_ = std::make_unique<VisualServoModule>(
      servo_gain_xy_,
      servo_gain_z_,
      servo_gain_yaw_,
      cnn_confidence_threshold_,
      max_translation_speed_,
      max_yaw_speed_deg_);
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
      std::bind(
        &PackeeArmController::PublishPoseStatus,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5,
        std::placeholders::_6),
      std::bind(
        &PackeeArmController::PublishPickStatus,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5,
        std::placeholders::_6,
        std::placeholders::_7,
        std::placeholders::_8),
      std::bind(
        &PackeeArmController::PublishPlaceStatus,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5,
        std::placeholders::_6,
        std::placeholders::_7,
        std::placeholders::_8),
      visual_servo_.get(),
      driver_.get(),
      gripper_.get(),
      progress_publish_interval_sec_,
      command_timeout_sec_,
      cart_view_preset_,
      standby_preset_);

    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&PackeeArmController::OnParametersUpdated, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Packee Arm Controller 노드가 초기화되었습니다.");
  }

private:
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

  void HandleMoveToPose(
    const std::shared_ptr<PackeeArmMoveToPose::Request> request,
    std::shared_ptr<PackeeArmMoveToPose::Response> response) {
    if (!valid_pose_types_.count(request->pose_type)) {
      PublishPoseStatus(
        request->robot_id,
        request->order_id,
        request->pose_type,
        "failed",
        0.0F,
        "지원되지 않는 pose_type입니다.");
      response->accepted = false;
      response->message = "유효하지 않은 pose_type입니다.";
      return;
    }

    MoveCommand command{};
    command.robot_id = request->robot_id;
    command.order_id = request->order_id;
    command.pose_type = request->pose_type;
    execution_manager_->EnqueueMove(command);
    response->accepted = true;
    response->message = "자세 변경 명령을 수락했습니다.";
    RCLCPP_INFO(
      this->get_logger(),
      "자세 변경 명령 수신: robot_id=%d, order_id=%d, pose_type=%s",
      request->robot_id,
      request->order_id,
      request->pose_type.c_str());
  }

  void HandlePickProduct(
    const std::shared_ptr<PackeeArmPickProduct::Request> request,
    std::shared_ptr<PackeeArmPickProduct::Response> response) {
    if (!valid_arm_sides_.count(request->arm_side)) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "arm_side가 유효하지 않습니다.");
      response->accepted = false;
      response->message = "arm_side가 유효하지 않습니다.";
      return;
    }

    if (!IsValidBoundingBox(
        request->bbox.x1,
        request->bbox.y1,
        request->bbox.x2,
        request->bbox.y2)) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "bbox 좌표가 유효하지 않습니다.");
      response->accepted = false;
      response->message = "bbox가 유효하지 않습니다.";
      return;
    }

    if (!AreFinite(
        request->target_position.x,
        request->target_position.y,
        request->target_position.z)) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "target_position에 유효하지 않은 값이 포함되어 있습니다.");
      response->accepted = false;
      response->message = "target_position 값이 잘못되었습니다.";
      return;
    }

    if (!IsWithinWorkspace(
        request->target_position.x,
        request->target_position.y,
        request->target_position.z)) {
      PublishPickStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "target_position이 myCobot 280 작업 공간을 벗어났습니다.");
      response->accepted = false;
      response->message = "target_position이 작업 공간 범위를 벗어났습니다.";
      return;
    }

    PickCommand command{};
    command.robot_id = request->robot_id;
    command.order_id = request->order_id;
    command.product_id = request->product_id;
    command.arm_side = request->arm_side;
    command.target_x = request->target_position.x;
    command.target_y = request->target_position.y;
    command.target_z = request->target_position.z;
    command.bbox_x1 = request->bbox.x1;
    command.bbox_y1 = request->bbox.y1;
    command.bbox_x2 = request->bbox.x2;
    command.bbox_y2 = request->bbox.y2;
    execution_manager_->EnqueuePick(command);
    response->accepted = true;
    response->message = "상품 픽업 명령을 수락했습니다.";
    RCLCPP_INFO(
      this->get_logger(),
      "픽업 명령 수신: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s",
      request->robot_id,
      request->order_id,
      request->product_id,
      request->arm_side.c_str());
  }

  void HandlePlaceProduct(
    const std::shared_ptr<PackeeArmPlaceProduct::Request> request,
    std::shared_ptr<PackeeArmPlaceProduct::Response> response) {
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
      response->accepted = false;
      response->message = "arm_side가 유효하지 않습니다.";
      return;
    }

    if (!AreFinite(
        request->box_position.x,
        request->box_position.y,
        request->box_position.z)) {
      PublishPlaceStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "box_position에 유효하지 않은 값이 포함되어 있습니다.");
      response->accepted = false;
      response->message = "box_position 값이 잘못되었습니다.";
      return;
    }

    if (!IsWithinWorkspace(
        request->box_position.x,
        request->box_position.y,
        request->box_position.z)) {
      PublishPlaceStatus(
        request->robot_id,
        request->order_id,
        request->product_id,
        request->arm_side,
        "failed",
        "planning",
        0.0F,
        "box_position이 myCobot 280 작업 공간을 벗어났습니다.");
      response->accepted = false;
      response->message = "box_position이 작업 공간 범위를 벗어났습니다.";
      return;
    }

    PlaceCommand command{};
    command.robot_id = request->robot_id;
    command.order_id = request->order_id;
    command.product_id = request->product_id;
    command.arm_side = request->arm_side;
    command.box_x = request->box_position.x;
    command.box_y = request->box_position.y;
    command.box_z = request->box_position.z;
    execution_manager_->EnqueuePlace(command);
    response->accepted = true;
    response->message = "상품 담기 명령을 수락했습니다.";
    RCLCPP_INFO(
      this->get_logger(),
      "담기 명령 수신: robot_id=%d, order_id=%d, product_id=%d, arm_side=%s",
      request->robot_id,
      request->order_id,
      request->product_id,
      request->arm_side.c_str());
  }

  void PublishPoseStatus(
    int32_t robot_id,
    int32_t order_id,
    const std::string & pose_type,
    const std::string & status,
    float progress,
    const std::string & message) {
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

  rcl_interfaces::msg::SetParametersResult OnParametersUpdated(
    const std::vector<rclcpp::Parameter> & parameters) {
    auto result = rcl_interfaces::msg::SetParametersResult();
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
      if ("servo_gain_xy" == parameter.get_name()) {
        next_servo_gain_xy = parameter.as_double();
      } else if ("servo_gain_z" == parameter.get_name()) {
        next_servo_gain_z = parameter.as_double();
      } else if ("servo_gain_yaw" == parameter.get_name()) {
        next_servo_gain_yaw = parameter.as_double();
      } else if ("cnn_confidence_threshold" == parameter.get_name()) {
        next_confidence_threshold = parameter.as_double();
      } else if ("max_translation_speed" == parameter.get_name()) {
        next_max_translation_speed = parameter.as_double();
      } else if ("max_yaw_speed_deg" == parameter.get_name()) {
        next_max_yaw_speed_deg = parameter.as_double();
      } else if ("gripper_force_limit" == parameter.get_name()) {
        next_gripper_force_limit = parameter.as_double();
      } else if ("progress_publish_interval" == parameter.get_name()) {
        next_progress_interval = parameter.as_double();
      } else if ("command_timeout_sec" == parameter.get_name()) {
        next_command_timeout = parameter.as_double();
      } else if ("preset_pose_cart_view" == parameter.get_name()) {
        next_cart_view_preset = ParsePoseParameter(
          parameter.as_double_array(),
          "preset_pose_cart_view",
          default_cart_view_pose_);
      } else if ("preset_pose_standby" == parameter.get_name()) {
        next_standby_preset = ParsePoseParameter(
          parameter.as_double_array(),
          "preset_pose_standby",
          default_standby_pose_);
      }
    }

    if (next_servo_gain_xy <= 0.0 || next_servo_gain_z <= 0.0 || next_servo_gain_yaw <= 0.0) {
      result.successful = false;
      result.reason = "서보 게인은 0보다 커야 합니다.";
      return result;
    }
    if (next_confidence_threshold <= 0.0 || next_confidence_threshold > 1.0) {
      result.successful = false;
      result.reason = "cnn_confidence_threshold는 (0, 1] 범위여야 합니다.";
      return result;
    }
    if (next_max_translation_speed <= 0.0 || next_max_yaw_speed_deg <= 0.0) {
      result.successful = false;
      result.reason = "속도 제한은 0보다 커야 합니다.";
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

  bool IsValidBoundingBox(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    return x2 > x1 && y2 > y1;
  }

  bool AreFinite(double x, double y, double z) const {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  rclcpp::Publisher<ArmPoseStatus>::SharedPtr pose_status_pub_;
  rclcpp::Publisher<PackeeArmTaskStatus>::SharedPtr pick_status_pub_;
  rclcpp::Publisher<PackeeArmTaskStatus>::SharedPtr place_status_pub_;

  rclcpp::Service<PackeeArmMoveToPose>::SharedPtr move_service_;
  rclcpp::Service<PackeeArmPickProduct>::SharedPtr pick_service_;
  rclcpp::Service<PackeeArmPlaceProduct>::SharedPtr place_service_;

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

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
  const std::array<double, 4> default_cart_view_pose_{{0.16, 0.0, 0.18, 0.0}};  // 카트 확인 자세
  const std::array<double, 4> default_standby_pose_{{0.10, 0.0, 0.14, 0.0}};  // 대기 자세
  PoseEstimate cart_view_preset_{};
  PoseEstimate standby_preset_{};
};

}  // namespace packee_arm

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<packee_arm::PackeeArmController>();
  try {
    rclcpp::spin(node);
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(node->get_logger(), "예외 발생: %s", exception.what());
  }
  rclcpp::shutdown();
  return 0;
}
