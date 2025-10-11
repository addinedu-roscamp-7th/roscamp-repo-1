// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PackeeArmTaskStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_arm_task_status.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ARM_TASK_STATUS__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ARM_TASK_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'arm_side'
// Member 'status'
// Member 'current_phase'
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/PackeeArmTaskStatus in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PackeeArmTaskStatus
{
  int32_t robot_id;
  int32_t order_id;
  int32_t product_id;
  rosidl_runtime_c__String arm_side;
  rosidl_runtime_c__String status;
  rosidl_runtime_c__String current_phase;
  float progress;
  rosidl_runtime_c__String message;
} shopee_interfaces__msg__PackeeArmTaskStatus;

// Struct for a sequence of shopee_interfaces__msg__PackeeArmTaskStatus.
typedef struct shopee_interfaces__msg__PackeeArmTaskStatus__Sequence
{
  shopee_interfaces__msg__PackeeArmTaskStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PackeeArmTaskStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ARM_TASK_STATUS__STRUCT_H_
