#ifndef PACKEE_ARM__PACKEE_ARM_CONTROLLER_HPP_
#define PACKEE_ARM__PACKEE_ARM_CONTROLLER_HPP_

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

struct PoseComponents {
  double x;
  double y;
  double z;
  double yaw_deg;
};

class PackeeArmController : public rclcpp::Node {
public:
  PackeeArmController();

private:
  void DeclareAndLoadParameters();
  PoseEstimate ParsePoseParameter(
    const std::vector<double> & values,
    const std::string & parameter_name,
    const std::array<double, 4> & fallback) const;
  PoseComponents ExtractPoseFromPoseMsg(const shopee_interfaces::msg::Pose6D & pose_msg) const;
  template<typename DetectedProductT>
  PoseComponents ExtractPoseFromDetectedProduct(const DetectedProductT & product) const;
  PoseEstimate MakePoseFromArray(const std::array<double, 4> & values) const;
  template<typename PoseT>
  PoseComponents ConvertPoseGeneric(const PoseT & pose) const;
  std::string NormalizePoseType(const std::string & pose_type) const;
  void HandleMoveToPose(
    const std::shared_ptr<ArmMoveToPose::Request> request,
    std::shared_ptr<ArmMoveToPose::Response> response);
  void HandlePickProduct(
    const std::shared_ptr<ArmPickProduct::Request> request,
    std::shared_ptr<ArmPickProduct::Response> response);
  void HandlePlaceProduct(
    const std::shared_ptr<ArmPlaceProduct::Request> request,
    std::shared_ptr<ArmPlaceProduct::Response> response);
  void PublishPoseStatus(int32_t robot_id, int32_t order_id,
                         const std::string &pose_type,
                         const std::string &status, float progress,
                         const std::string &message);
  void PublishPickStatus(int32_t robot_id, int32_t order_id, int32_t product_id,
                         const std::string &arm_side,
                         const std::string &status,
                         const std::string &phase,
                         float progress, const std::string &message);
  void PublishPlaceStatus(int32_t robot_id, int32_t order_id, int32_t product_id,
                          const std::string &arm_side,
                          const std::string &status,
                          const std::string &phase,
                          float progress, const std::string &message);
  rcl_interfaces::msg::SetParametersResult OnParametersUpdated(
    const std::vector<rclcpp::Parameter> & parameters);
  bool AreFinite(double x, double y, double z) const;
  bool AreFinite(const PoseComponents & pose) const;
  bool IsZeroPose(const PoseComponents & pose) const;
  bool IsWithinWorkspace(double x, double y, double z) const;
  void ClampPoseToWorkspace(PoseComponents * pose) const;
  bool IsValidBoundingBox(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const;

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

  double servo_gain_xy_;
  double servo_gain_z_;
  double servo_gain_yaw_;
  double cnn_confidence_threshold_;
  double max_translation_speed_;
  double max_yaw_speed_deg_;
  double gripper_force_limit_;
  double progress_publish_interval_sec_;
  double command_timeout_sec_;
  std::string left_velocity_topic_;
  std::string right_velocity_topic_;
  std::string velocity_frame_id_;
  std::string left_gripper_topic_;
  std::string right_gripper_topic_;
  const std::array<double, 4> default_cart_view_pose_{{0.16, 0.0, 0.18, 0.0}};
  const std::array<double, 4> default_standby_pose_{{0.10, 0.0, 0.14, 0.0}};
  PoseEstimate cart_view_preset_{};
  PoseEstimate standby_preset_{};
};

}  // namespace packee_arm

#endif  // PACKEE_ARM__PACKEE_ARM_CONTROLLER_HPP_