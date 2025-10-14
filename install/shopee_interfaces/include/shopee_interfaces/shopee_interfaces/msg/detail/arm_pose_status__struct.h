// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/ArmPoseStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/arm_pose_status.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__ARM_POSE_STATUS__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__ARM_POSE_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'pose_type'
// Member 'status'
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/ArmPoseStatus in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__ArmPoseStatus
{
  int32_t robot_id;
  int32_t order_id;
  rosidl_runtime_c__String pose_type;
  rosidl_runtime_c__String status;
  float progress;
  rosidl_runtime_c__String message;
} shopee_interfaces__msg__ArmPoseStatus;

// Struct for a sequence of shopee_interfaces__msg__ArmPoseStatus.
typedef struct shopee_interfaces__msg__ArmPoseStatus__Sequence
{
  shopee_interfaces__msg__ArmPoseStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__ArmPoseStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__ARM_POSE_STATUS__STRUCT_H_
