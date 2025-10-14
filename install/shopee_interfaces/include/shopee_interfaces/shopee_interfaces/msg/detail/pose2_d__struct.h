// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/Pose2D.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pose2_d.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__POSE2_D__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__POSE2_D__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/Pose2D in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__Pose2D
{
  float x;
  float y;
  float theta;
} shopee_interfaces__msg__Pose2D;

// Struct for a sequence of shopee_interfaces__msg__Pose2D.
typedef struct shopee_interfaces__msg__Pose2D__Sequence
{
  shopee_interfaces__msg__Pose2D * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__Pose2D__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__POSE2_D__STRUCT_H_
