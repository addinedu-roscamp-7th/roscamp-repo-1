#ifndef PACKEE_ARM_EXECUTION_MANAGER_HPP_
#define PACKEE_ARM_EXECUTION_MANAGER_HPP_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "rclcpp/logger.hpp"
#include "rclcpp/node.hpp"

#include "packee_arm/arm_driver_proxy.hpp"
#include "packee_arm/constants.hpp"
#include "packee_arm/gripper_controller.hpp"
#include "packee_arm/types.hpp"
#include "packee_arm/visual_servo_module.hpp"

namespace packee_arm {

// ExecutionManager 클래스는 좌/우 팔 명령 큐와 시각 서보 실행을 관리한다.
class ExecutionManager {
public:
  using PoseStatusCallback = std::function<
    void(int32_t, int32_t, const std::string &, const std::string &, float, const std::string &)>;
  using TaskStatusCallback = std::function<
    void(int32_t, int32_t, int32_t, const std::string &, const std::string &, const std::string &, float, const std::string &)>;

  ExecutionManager(
    rclcpp::Node * node,
    PoseStatusCallback pose_callback,
    TaskStatusCallback pick_callback,
    TaskStatusCallback place_callback,
    VisualServoModule * visual_servo,
    ArmDriverProxy * driver,
    GripperController * gripper,
    double progress_interval_sec,
    double command_timeout_sec,
    const PoseEstimate & cart_view_preset,
    const PoseEstimate & standby_preset);

  ~ExecutionManager();

  bool EnqueueMove(const MoveCommand & command);

  bool EnqueuePick(const PickCommand & command);

  bool EnqueuePlace(const PlaceCommand & command);

  bool IsArmBusy(const std::string & arm_side);

  bool IsHoldingProduct(const std::string & arm_side);

  void UpdateTiming(double progress_interval_sec, double command_timeout_sec);

  // 런타임에 myCobot 280 프리셋 자세를 갱신한다.
  void UpdatePosePresets(
    const PoseEstimate & cart_view_preset,
    const PoseEstimate & standby_preset);

private:
  struct ArmState {
    PoseEstimate pose{};
    bool holding_product{false};
    std::optional<int32_t> product_id{};
  };

  struct ArmWorkItem {
    ArmCommandType type;
    PickCommand pick;
    PlaceCommand place;
  };

  bool EnqueueArmCommand(
    std::mutex & mutex,
    std::condition_variable & cv,
    std::queue<ArmWorkItem> & queue,
    const ArmWorkItem & item);

  bool HasArmState(const std::string & arm_side);

  void RunMoveWorker();

  void RunArmWorker(const std::string & arm_side);

  void ProcessMoveCommand(const MoveCommand & command);

  void ProcessPickCommand(const PickCommand & command);

  void ProcessPlaceCommand(const PlaceCommand & command);

  bool DriveServo(
    const std::string & arm_side,
    const PoseEstimate & target_pose,
    float progress_start,
    float progress_end,
    const std::function<void(float, const std::string &)> & progress_callback,
    std::string * failure_reason);

  PoseEstimate GetArmPose(const std::string & arm_side);

  void UpdateArmPose(const std::string & arm_side, const PoseEstimate & pose);

  void UpdateArmHoldingState(
    const std::string & arm_side,
    bool holding,
    std::optional<int32_t> product_id);

  bool IsArmReadyToPlace(const std::string & arm_side, int32_t product_id);

  std::chrono::milliseconds DurationFromInterval() const;

  // 런치 파라미터로 주어진 프리셋 자세를 pose_type에 맞게 반환한다.
  PoseEstimate GetPresetPose(const std::string & pose_type) const;

  void ApplyPoseToArms(const PoseEstimate & pose);

  double CalculateBoundingBoxConfidence(
    int32_t x1,
    int32_t y1,
    int32_t x2,
    int32_t y2) const;

  std::queue<ArmWorkItem> & GetArmQueue(const std::string & arm_side);

  std::mutex & GetArmMutex(const std::string & arm_side);

  rclcpp::Node * node_;
  rclcpp::Logger logger_;
  PoseStatusCallback pose_callback_;
  TaskStatusCallback pick_callback_;
  TaskStatusCallback place_callback_;
  VisualServoModule * visual_servo_;
  ArmDriverProxy * driver_;
  GripperController * gripper_;
  double progress_interval_sec_;
  double command_timeout_sec_;
  std::atomic<bool> running_;
  mutable std::mutex preset_mutex_;
  PoseEstimate cart_view_preset_;
  PoseEstimate standby_preset_;

  std::mutex move_mutex_;
  std::condition_variable move_cv_;
  std::queue<MoveCommand> move_queue_;
  std::thread move_thread_;

  std::mutex left_mutex_;
  std::condition_variable left_cv_;
  std::queue<ArmWorkItem> left_queue_;
  std::thread left_thread_;
  std::atomic<bool> left_processing_{false};

  std::mutex right_mutex_;
  std::condition_variable right_cv_;
  std::queue<ArmWorkItem> right_queue_;
  std::thread right_thread_;
  std::atomic<bool> right_processing_{false};

  std::mutex arm_state_mutex_;
  std::unordered_map<std::string, ArmState> arm_states_;
};

}  // namespace packee_arm

#endif  // PACKEE_ARM_EXECUTION_MANAGER_HPP_
