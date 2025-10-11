// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeMobileArrival.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_arrival.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'final_pose'
#include "shopee_interfaces/msg/detail/pose2_d__struct.h"
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/PickeeMobileArrival in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeMobileArrival
{
  int32_t robot_id;
  int32_t order_id;
  int32_t location_id;
  shopee_interfaces__msg__Pose2D final_pose;
  float position_error;
  float travel_time;
  rosidl_runtime_c__String message;
} shopee_interfaces__msg__PickeeMobileArrival;

// Struct for a sequence of shopee_interfaces__msg__PickeeMobileArrival.
typedef struct shopee_interfaces__msg__PickeeMobileArrival__Sequence
{
  shopee_interfaces__msg__PickeeMobileArrival * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeMobileArrival__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__STRUCT_H_
