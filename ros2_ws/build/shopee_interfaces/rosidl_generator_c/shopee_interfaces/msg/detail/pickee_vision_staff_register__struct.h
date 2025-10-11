// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeVisionStaffRegister.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_staff_register.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/PickeeVisionStaffRegister in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeVisionStaffRegister
{
  int32_t robot_id;
  bool success;
  rosidl_runtime_c__String message;
} shopee_interfaces__msg__PickeeVisionStaffRegister;

// Struct for a sequence of shopee_interfaces__msg__PickeeVisionStaffRegister.
typedef struct shopee_interfaces__msg__PickeeVisionStaffRegister__Sequence
{
  shopee_interfaces__msg__PickeeVisionStaffRegister * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeVisionStaffRegister__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__STRUCT_H_
