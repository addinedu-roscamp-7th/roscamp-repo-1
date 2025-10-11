// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PackeeRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_robot_status.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'state'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/PackeeRobotStatus in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PackeeRobotStatus
{
  int32_t robot_id;
  rosidl_runtime_c__String state;
  int32_t current_order_id;
  int32_t items_in_cart;
} shopee_interfaces__msg__PackeeRobotStatus;

// Struct for a sequence of shopee_interfaces__msg__PackeeRobotStatus.
typedef struct shopee_interfaces__msg__PackeeRobotStatus__Sequence
{
  shopee_interfaces__msg__PackeeRobotStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PackeeRobotStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__STRUCT_H_
