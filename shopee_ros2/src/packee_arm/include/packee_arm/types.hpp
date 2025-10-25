#ifndef PACKEE_ARM_TYPES_HPP_
#define PACKEE_ARM_TYPES_HPP_

#include <cstdint>
#include <optional>
#include <string>

namespace packee_arm {

// PoseEstimate 구조체는 시각 서보 목표/현재 포즈 정보를 담는다.
struct PoseEstimate {
  double x;
  double y;
  double z;
  double yaw_deg;
  double confidence;
};

// ArmCommandType 열거형은 작업 큐에 저장되는 명령 타입을 나타낸다.
enum class ArmCommandType {
  MoveToPose,
  PickProduct,
  PlaceProduct
};

// MoveCommand 구조체는 자세 변경 명령 입력을 저장한다.
struct MoveCommand {
  int32_t robot_id;
  int32_t order_id;
  std::string pose_type;
};

// PickCommand 구조체는 상품 픽업 명령에 필요한 정보를 담는다.
struct PickCommand {
  int32_t robot_id;
  int32_t order_id;
  int32_t product_id;
  std::string arm_side;
  PoseEstimate target_pose;
  double detection_confidence;
  int32_t bbox_x1;
  int32_t bbox_y1;
  int32_t bbox_x2;
  int32_t bbox_y2;
};

// PlaceCommand 구조체는 상품 담기 명령에 필요한 정보를 담는다.
struct PlaceCommand {
  int32_t robot_id;
  int32_t order_id;
  int32_t product_id;
  std::string arm_side;
  PoseEstimate target_pose;
};

}  // namespace packee_arm

#endif  // PACKEE_ARM_TYPES_HPP_
