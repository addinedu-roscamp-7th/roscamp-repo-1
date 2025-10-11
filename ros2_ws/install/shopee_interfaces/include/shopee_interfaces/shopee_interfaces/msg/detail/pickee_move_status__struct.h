// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeMoveStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_move_status.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOVE_STATUS__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOVE_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/PickeeMoveStatus in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeMoveStatus
{
  int32_t robot_id;
  int32_t order_id;
  int32_t location_id;
} shopee_interfaces__msg__PickeeMoveStatus;

// Struct for a sequence of shopee_interfaces__msg__PickeeMoveStatus.
typedef struct shopee_interfaces__msg__PickeeMoveStatus__Sequence
{
  shopee_interfaces__msg__PickeeMoveStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeMoveStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOVE_STATUS__STRUCT_H_
