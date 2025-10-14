// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeProductLoaded.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_product_loaded.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_LOADED__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_LOADED__STRUCT_H_

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

/// Struct defined in msg/PickeeProductLoaded in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeProductLoaded
{
  int32_t robot_id;
  int32_t product_id;
  int32_t quantity;
  bool success;
  rosidl_runtime_c__String message;
} shopee_interfaces__msg__PickeeProductLoaded;

// Struct for a sequence of shopee_interfaces__msg__PickeeProductLoaded.
typedef struct shopee_interfaces__msg__PickeeProductLoaded__Sequence
{
  shopee_interfaces__msg__PickeeProductLoaded * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeProductLoaded__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_LOADED__STRUCT_H_
