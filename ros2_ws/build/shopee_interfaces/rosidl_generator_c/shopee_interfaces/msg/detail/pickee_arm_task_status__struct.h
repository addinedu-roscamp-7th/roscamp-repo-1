// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeArmTaskStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_arm_task_status.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARM_TASK_STATUS__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARM_TASK_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'status'
// Member 'current_phase'
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/PickeeArmTaskStatus in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeArmTaskStatus
{
  int32_t robot_id;
  int32_t order_id;
  int32_t product_id;
  rosidl_runtime_c__String status;
  rosidl_runtime_c__String current_phase;
  float progress;
  rosidl_runtime_c__String message;
} shopee_interfaces__msg__PickeeArmTaskStatus;

// Struct for a sequence of shopee_interfaces__msg__PickeeArmTaskStatus.
typedef struct shopee_interfaces__msg__PickeeArmTaskStatus__Sequence
{
  shopee_interfaces__msg__PickeeArmTaskStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeArmTaskStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARM_TASK_STATUS__STRUCT_H_
