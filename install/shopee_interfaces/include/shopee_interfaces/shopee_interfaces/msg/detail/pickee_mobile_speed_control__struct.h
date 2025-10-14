// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeMobileSpeedControl.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_speed_control.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'speed_mode'
// Member 'reason'
#include "rosidl_runtime_c/string.h"
// Member 'obstacles'
#include "shopee_interfaces/msg/detail/obstacle__struct.h"

/// Struct defined in msg/PickeeMobileSpeedControl in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeMobileSpeedControl
{
  int32_t robot_id;
  int32_t order_id;
  rosidl_runtime_c__String speed_mode;
  float target_speed;
  shopee_interfaces__msg__Obstacle__Sequence obstacles;
  rosidl_runtime_c__String reason;
} shopee_interfaces__msg__PickeeMobileSpeedControl;

// Struct for a sequence of shopee_interfaces__msg__PickeeMobileSpeedControl.
typedef struct shopee_interfaces__msg__PickeeMobileSpeedControl__Sequence
{
  shopee_interfaces__msg__PickeeMobileSpeedControl * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeMobileSpeedControl__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__STRUCT_H_
