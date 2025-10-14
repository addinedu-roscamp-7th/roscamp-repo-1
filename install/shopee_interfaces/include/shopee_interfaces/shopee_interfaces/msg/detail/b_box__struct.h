// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/BBox.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/b_box.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/BBox in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__BBox
{
  int32_t x1;
  int32_t y1;
  int32_t x2;
  int32_t y2;
} shopee_interfaces__msg__BBox;

// Struct for a sequence of shopee_interfaces__msg__BBox.
typedef struct shopee_interfaces__msg__BBox__Sequence
{
  shopee_interfaces__msg__BBox * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__BBox__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__STRUCT_H_
