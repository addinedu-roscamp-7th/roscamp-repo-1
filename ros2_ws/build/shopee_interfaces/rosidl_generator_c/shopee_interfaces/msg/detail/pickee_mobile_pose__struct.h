// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeMobilePose.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_pose.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_POSE__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_POSE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'current_pose'
#include "shopee_interfaces/msg/detail/pose2_d__struct.h"
// Member 'status'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/PickeeMobilePose in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeMobilePose
{
  int32_t robot_id;
  int32_t order_id;
  shopee_interfaces__msg__Pose2D current_pose;
  float linear_velocity;
  float angular_velocity;
  float battery_level;
  rosidl_runtime_c__String status;
} shopee_interfaces__msg__PickeeMobilePose;

// Struct for a sequence of shopee_interfaces__msg__PickeeMobilePose.
typedef struct shopee_interfaces__msg__PickeeMobilePose__Sequence
{
  shopee_interfaces__msg__PickeeMobilePose * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeMobilePose__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_POSE__STRUCT_H_
