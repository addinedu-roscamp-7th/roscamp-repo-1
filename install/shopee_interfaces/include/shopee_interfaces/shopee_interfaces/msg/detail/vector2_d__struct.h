// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/Vector2D.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/vector2_d.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/Vector2D in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__Vector2D
{
  float vx;
  float vy;
} shopee_interfaces__msg__Vector2D;

// Struct for a sequence of shopee_interfaces__msg__Vector2D.
typedef struct shopee_interfaces__msg__Vector2D__Sequence
{
  shopee_interfaces__msg__Vector2D * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__Vector2D__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__STRUCT_H_
