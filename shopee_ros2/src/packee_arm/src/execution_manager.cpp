#include "packee_arm/execution_manager.hpp"

#include <algorithm>
#include <cmath>

namespace packee_arm {

ExecutionManager::ExecutionManager(
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
  const PoseEstimate & standby_preset)
: node_(node),
  logger_(node_->get_logger()),
  pose_callback_(std::move(pose_callback)),
  pick_callback_(std::move(pick_callback)),
  place_callback_(std::move(place_callback)),
  visual_servo_(visual_servo),
  driver_(driver),
  gripper_(gripper),
  progress_interval_sec_(progress_interval_sec),
  command_timeout_sec_(command_timeout_sec),
  running_(true),
  cart_view_preset_(cart_view_preset),
  standby_preset_(standby_preset) {
  // 좌/우 팔 상태를 초기화하고 전용 작업 스레드를 기동한다.
  ArmState left_state{};
  ArmState right_state{};
  arm_states_.emplace("left", left_state);
  arm_states_.emplace("right", right_state);
  move_thread_ = std::thread(&ExecutionManager::RunMoveWorker, this);
  left_thread_ = std::thread(&ExecutionManager::RunArmWorker, this, "left");
  right_thread_ = std::thread(&ExecutionManager::RunArmWorker, this, "right");
}


ExecutionManager::~ExecutionManager() {
  running_.store(false);
  move_cv_.notify_all();
  left_cv_.notify_all();
  right_cv_.notify_all();
  if (move_thread_.joinable()) {
    move_thread_.join();
  }
  if (left_thread_.joinable()) {
    left_thread_.join();
  }
  if (right_thread_.joinable()) {
    right_thread_.join();
  }
}


bool ExecutionManager::EnqueueMove(const MoveCommand & command) {
  {
    std::lock_guard<std::mutex> lock(move_mutex_);
    move_queue_.push(command);
  }
  move_cv_.notify_one();
  return true;
}


bool ExecutionManager::EnqueuePick(const PickCommand & command) {
  return EnqueueArmCommand(
    "left" == command.arm_side ? left_mutex_ : right_mutex_,
    "left" == command.arm_side ? left_cv_ : right_cv_,
    "left" == command.arm_side ? left_queue_ : right_queue_,
    ArmWorkItem{ArmCommandType::PickProduct, command, PlaceCommand{}});
}


bool ExecutionManager::EnqueuePlace(const PlaceCommand & command) {
  return EnqueueArmCommand(
    "left" == command.arm_side ? left_mutex_ : right_mutex_,
    "left" == command.arm_side ? left_cv_ : right_cv_,
    "left" == command.arm_side ? left_queue_ : right_queue_,
    ArmWorkItem{ArmCommandType::PlaceProduct, PickCommand{}, command});
}


bool ExecutionManager::IsArmBusy(const std::string & arm_side) {
  if (!HasArmState(arm_side)) {
    return true;
  }
  std::lock_guard<std::mutex> lock(GetArmMutex(arm_side));
  auto & queue = GetArmQueue(arm_side);
  const bool processing = "left" == arm_side ? left_processing_.load() : right_processing_.load();
  return processing || !queue.empty();
}


bool ExecutionManager::IsHoldingProduct(const std::string & arm_side) {
  std::lock_guard<std::mutex> lock(arm_state_mutex_);
  if (!arm_states_.count(arm_side)) {
    return false;
  }
  return arm_states_.at(arm_side).holding_product;
}


void ExecutionManager::UpdateTiming(double progress_interval_sec, double command_timeout_sec) {
  progress_interval_sec_ = progress_interval_sec;
  command_timeout_sec_ = command_timeout_sec;
}


void ExecutionManager::UpdatePosePresets(
  const PoseEstimate & cart_view_preset,
  const PoseEstimate & standby_preset) {
  // myCobot 280 듀얼 암 기준 자세를 동적으로 교체한다.
  std::lock_guard<std::mutex> lock(preset_mutex_);
  cart_view_preset_ = cart_view_preset;
  standby_preset_ = standby_preset;
}


bool ExecutionManager::EnqueueArmCommand(
  std::mutex & mutex,
  std::condition_variable & cv,
  std::queue<ArmWorkItem> & queue,
  const ArmWorkItem & item) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(item);
  }
  cv.notify_one();
  return true;
}


bool ExecutionManager::HasArmState(const std::string & arm_side) {
  std::lock_guard<std::mutex> lock(arm_state_mutex_);
  return arm_states_.count(arm_side) > 0;
}


void ExecutionManager::RunMoveWorker() {
  while (running_.load()) {
    MoveCommand command{};
    {
      std::unique_lock<std::mutex> lock(move_mutex_);
      move_cv_.wait(lock, [this]() {
        return !move_queue_.empty() || !running_.load();
      });
      if (!running_.load() && move_queue_.empty()) {
        return;
      }
      command = move_queue_.front();
      move_queue_.pop();
    }
    ProcessMoveCommand(command);
  }
}


void ExecutionManager::RunArmWorker(const std::string & arm_side) {
  std::queue<ArmWorkItem> * queue_ptr = nullptr;
  std::condition_variable * cv_ptr = nullptr;
  std::mutex * mutex_ptr = nullptr;
  std::atomic<bool> * processing_ptr = nullptr;
  if ("left" == arm_side) {
    queue_ptr = &left_queue_;
    cv_ptr = &left_cv_;
    mutex_ptr = &left_mutex_;
    processing_ptr = &left_processing_;
  } else {
    queue_ptr = &right_queue_;
    cv_ptr = &right_cv_;
    mutex_ptr = &right_mutex_;
    processing_ptr = &right_processing_;
  }

  while (running_.load()) {
    ArmWorkItem item{ArmCommandType::MoveToPose, PickCommand{}, PlaceCommand{}};
    {
      std::unique_lock<std::mutex> lock(*mutex_ptr);
      cv_ptr->wait(lock, [&]() {
        return !queue_ptr->empty() || !running_.load();
      });
      if (!running_.load() && queue_ptr->empty()) {
        return;
      }
      item = queue_ptr->front();
      queue_ptr->pop();
      processing_ptr->store(true);
    }

    if (ArmCommandType::PickProduct == item.type) {
      ProcessPickCommand(item.pick);
    } else if (ArmCommandType::PlaceProduct == item.type) {
      ProcessPlaceCommand(item.place);
    }

    processing_ptr->store(false);
  }
}


void ExecutionManager::ProcessMoveCommand(const MoveCommand & command) {
  pose_callback_(
    command.robot_id,
    command.order_id,
    command.pose_type,
    "in_progress",
    0.1F,
    "자세 변경 명령을 수락했습니다.");
  const auto target_pose = GetPresetPose(command.pose_type);
  ApplyPoseToArms(target_pose);
  std::this_thread::sleep_for(DurationFromInterval());
  pose_callback_(
    command.robot_id,
    command.order_id,
    command.pose_type,
    "in_progress",
    0.6F,
    "자세 이동을 수행 중입니다.");
  std::this_thread::sleep_for(DurationFromInterval());
  pose_callback_(
    command.robot_id,
    command.order_id,
    command.pose_type,
    "completed",
    1.0F,
    "자세 변경을 완료했습니다.");
}


void ExecutionManager::ProcessPickCommand(const PickCommand & command) {
  const std::string & arm_side = command.arm_side;
  const double confidence = CalculateBoundingBoxConfidence(
    command.bbox_x1,
    command.bbox_y1,
    command.bbox_x2,
    command.bbox_y2);
  if (confidence < visual_servo_->GetConfidenceThreshold()) {
    pick_callback_(
      command.robot_id,
      command.order_id,
      command.product_id,
      arm_side,
      "failed",
      "servoing",
      0.0F,
      "CNN 신뢰도가 임계값보다 낮습니다. 재탐지 요청이 필요합니다.");
    return;
  }

  pick_callback_(
    command.robot_id,
    command.order_id,
    command.product_id,
    arm_side,
    "in_progress",
    "servoing",
    0.15F,
    "시각 서보 정렬을 준비 중입니다.");

  PoseEstimate target_pose{};
  target_pose.x = command.target_x;
  target_pose.y = command.target_y;
  target_pose.z = command.target_z;
  target_pose.yaw_deg = command.target_yaw_deg;
  target_pose.confidence = confidence;

  std::string failure_reason;
  const bool servo_success = DriveServo(
    arm_side,
    target_pose,
    0.15F,
    0.65F,
    [&](float progress, const std::string & detail) {
      pick_callback_(
        command.robot_id,
        command.order_id,
        command.product_id,
        arm_side,
        "in_progress",
        "servoing",
        progress,
        detail);
    },
    &failure_reason);

  if (!servo_success) {
    pick_callback_(
      command.robot_id,
      command.order_id,
      command.product_id,
      arm_side,
      "failed",
      "servoing",
      0.3F,
      failure_reason);
    return;
  }

  pick_callback_(
    command.robot_id,
    command.order_id,
    command.product_id,
    arm_side,
    "in_progress",
    "grasping",
    0.75F,
    "상품을 파지하고 있습니다.");

  const double grip_force = std::min(0.9 * gripper_->GetForceLimit(), 30.0);
  if (!gripper_->Close(arm_side, grip_force)) {
    pick_callback_(
      command.robot_id,
      command.order_id,
      command.product_id,
      arm_side,
      "failed",
      "grasping",
      0.75F,
      "그리퍼 힘 제한으로 인해 파지에 실패했습니다.");
    return;
  }

  UpdateArmHoldingState(arm_side, true, command.product_id);
  std::this_thread::sleep_for(DurationFromInterval());

  pick_callback_(
    command.robot_id,
    command.order_id,
    command.product_id,
    arm_side,
    "in_progress",
    "lifting",
    0.9F,
    "상품을 들어올리는 중입니다.");
  std::this_thread::sleep_for(DurationFromInterval());
  pick_callback_(
    command.robot_id,
    command.order_id,
    command.product_id,
    arm_side,
    "completed",
    "done",
    1.0F,
    "상품 픽업을 완료했습니다.");
}


void ExecutionManager::ProcessPlaceCommand(const PlaceCommand & command) {
  const std::string & arm_side = command.arm_side;
  if (!IsArmReadyToPlace(arm_side, command.product_id)) {
    place_callback_(
      command.robot_id,
      command.order_id,
      command.product_id,
      arm_side,
      "failed",
      "servoing",
      0.0F,
      "해당 팔이 상품을 보유하고 있지 않습니다.");
    return;
  }

  place_callback_(
    command.robot_id,
    command.order_id,
    command.product_id,
    arm_side,
    "in_progress",
    "servoing",
    0.1F,
    "포장 위치로 시각 서보 제어를 준비 중입니다.");

  PoseEstimate target_pose{};
  target_pose.x = command.box_x;
  target_pose.y = command.box_y;
  target_pose.z = command.box_z;
  target_pose.yaw_deg = command.box_yaw_deg;
  target_pose.confidence = 1.0;

  std::string failure_reason;
  const bool servo_success = DriveServo(
    arm_side,
    target_pose,
    0.1F,
    0.6F,
    [&](float progress, const std::string & detail) {
      place_callback_(
        command.robot_id,
        command.order_id,
        command.product_id,
        arm_side,
        "in_progress",
        "servoing",
        progress,
        detail);
    },
    &failure_reason);

  if (!servo_success) {
    place_callback_(
      command.robot_id,
      command.order_id,
      command.product_id,
      arm_side,
      "failed",
      "servoing",
      0.3F,
      failure_reason);
    return;
  }

  place_callback_(
    command.robot_id,
    command.order_id,
    command.product_id,
    arm_side,
    "in_progress",
    "placing",
    0.75F,
    "포장 박스 내부에 상품을 정렬 중입니다.");
  std::this_thread::sleep_for(DurationFromInterval());

  if (!gripper_->Open(arm_side)) {
    place_callback_(
      command.robot_id,
      command.order_id,
      command.product_id,
      arm_side,
      "failed",
      "placing",
      0.8F,
      "그리퍼 해제가 실패했습니다.");
    return;
  }

  UpdateArmHoldingState(arm_side, false, std::nullopt);
  std::this_thread::sleep_for(DurationFromInterval());
  place_callback_(
    command.robot_id,
    command.order_id,
    command.product_id,
    arm_side,
    "in_progress",
    "retreat",
    0.9F,
    "안전 위치로 복귀 중입니다.");
  std::this_thread::sleep_for(DurationFromInterval());
  place_callback_(
    command.robot_id,
    command.order_id,
    command.product_id,
    arm_side,
    "completed",
    "done",
    1.0F,
    "상품 담기를 완료했습니다.");
}


bool ExecutionManager::DriveServo(
  const std::string & arm_side,
  const PoseEstimate & target_pose,
  float progress_start,
  float progress_end,
  const std::function<void(float, const std::string &)> & progress_callback,
  std::string * failure_reason) {
  // myCobot 280 엔드이펙터가 목표 자세에 수렴할 때까지 P 제어 루프를 수행한다.
  const int max_steps = std::min(
    kMaxServoSteps,
    std::max(1, static_cast<int>(std::ceil(command_timeout_sec_ / progress_interval_sec_))));
  PoseEstimate current_pose = GetArmPose(arm_side);
  for (int step = 0; step < max_steps; ++step) {
    const auto servo_command = visual_servo_->ComputeCommand(current_pose, target_pose);
    if (servo_command.confidence < visual_servo_->GetConfidenceThreshold()) {
      *failure_reason = "시각 서보 신뢰도가 임계값 미만입니다.";
      return false;
    }
    if (servo_command.goal_reached) {
      UpdateArmPose(arm_side, target_pose);
      progress_callback(progress_end, "목표 자세에 도달했습니다.");
      return true;
    }
    if (!driver_->SendVelocityCommand(
        arm_side,
        servo_command.vx,
        servo_command.vy,
        servo_command.vz,
        servo_command.yaw_rate_deg)) {
      *failure_reason = "하드웨어 인터페이스가 속도 명령을 거부했습니다.";
      return false;
    }
    current_pose.x += servo_command.vx * progress_interval_sec_;
    current_pose.y += servo_command.vy * progress_interval_sec_;
    current_pose.z += servo_command.vz * progress_interval_sec_;
    current_pose.yaw_deg += servo_command.yaw_rate_deg * progress_interval_sec_;
    UpdateArmPose(arm_side, current_pose);

    const float alpha = static_cast<float>(step + 1) / static_cast<float>(max_steps);
    const float progress = progress_start + ((progress_end - progress_start) * alpha);
    progress_callback(progress, "시각 서보 제어를 수행 중입니다.");
    std::this_thread::sleep_for(DurationFromInterval());
  }
  *failure_reason = "시각 서보가 제한 시간 내에 수렴하지 못했습니다.";
  return false;
}


PoseEstimate ExecutionManager::GetArmPose(const std::string & arm_side) {
  std::lock_guard<std::mutex> lock(arm_state_mutex_);
  return arm_states_.at(arm_side).pose;
}


void ExecutionManager::UpdateArmPose(const std::string & arm_side, const PoseEstimate & pose) {
  std::lock_guard<std::mutex> lock(arm_state_mutex_);
  arm_states_.at(arm_side).pose = pose;
}


void ExecutionManager::UpdateArmHoldingState(
  const std::string & arm_side,
  bool holding,
  std::optional<int32_t> product_id) {
  std::lock_guard<std::mutex> lock(arm_state_mutex_);
  auto & state = arm_states_.at(arm_side);
  state.holding_product = holding;
  state.product_id = product_id;
}


bool ExecutionManager::IsArmReadyToPlace(const std::string & arm_side, int32_t product_id) {
  std::lock_guard<std::mutex> lock(arm_state_mutex_);
  if (!arm_states_.count(arm_side)) {
    return false;
  }
  const auto & state = arm_states_.at(arm_side);
  return state.holding_product && state.product_id.has_value() &&
         state.product_id.value() == product_id;
}


std::chrono::milliseconds ExecutionManager::DurationFromInterval() const {
  const int milliseconds = static_cast<int>(progress_interval_sec_ * 1000.0);
  return std::chrono::milliseconds(std::max(50, milliseconds));
}


PoseEstimate ExecutionManager::GetPresetPose(const std::string & pose_type) const {
  std::lock_guard<std::mutex> lock(preset_mutex_);
  PoseEstimate pose{};
  if ("cart_view" == pose_type) {
    pose = cart_view_preset_;
  } else {
    pose = standby_preset_;
  }
  // 프리셋은 이미 유효성 검증을 거쳤으므로 그대로 반환한다.
  return pose;
}


void ExecutionManager::ApplyPoseToArms(const PoseEstimate & pose) {
  UpdateArmPose("left", pose);
  UpdateArmPose("right", pose);
}


double ExecutionManager::CalculateBoundingBoxConfidence(
  int32_t x1,
  int32_t y1,
  int32_t x2,
  int32_t y2) const {
  if (x2 <= x1 || y2 <= y1) {
    return 0.0;
  }
  const double width = static_cast<double>(x2 - x1);
  const double height = static_cast<double>(y2 - y1);
  const double area = width * height;
  const double normalized =
    std::clamp(area / kBoundingBoxReferenceArea, 0.0, 1.0);
  return normalized;
}


std::queue<ExecutionManager::ArmWorkItem> & ExecutionManager::GetArmQueue(
  const std::string & arm_side) {
  if ("left" == arm_side) {
    return left_queue_;
  }
  return right_queue_;
}


std::mutex & ExecutionManager::GetArmMutex(const std::string & arm_side) {
  if ("left" == arm_side) {
    return left_mutex_;
  }
  return right_mutex_;
}

}  // namespace packee_arm
