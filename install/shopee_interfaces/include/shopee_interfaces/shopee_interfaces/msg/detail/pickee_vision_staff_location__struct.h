// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeVisionStaffLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_staff_location.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'relative_position'
#include "shopee_interfaces/msg/detail/point2_d__struct.h"

/// Struct defined in msg/PickeeVisionStaffLocation in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeVisionStaffLocation
{
  int32_t robot_id;
  shopee_interfaces__msg__Point2D relative_position;
  float distance;
  bool is_tracking;
} shopee_interfaces__msg__PickeeVisionStaffLocation;

// Struct for a sequence of shopee_interfaces__msg__PickeeVisionStaffLocation.
typedef struct shopee_interfaces__msg__PickeeVisionStaffLocation__Sequence
{
  shopee_interfaces__msg__PickeeVisionStaffLocation * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeVisionStaffLocation__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__STRUCT_H_
