// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/Point3D.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/point3_d.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__POINT3_D__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__POINT3_D__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/Point3D in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__Point3D
{
  float x;
  float y;
  float z;
} shopee_interfaces__msg__Point3D;

// Struct for a sequence of shopee_interfaces__msg__Point3D.
typedef struct shopee_interfaces__msg__Point3D__Sequence
{
  shopee_interfaces__msg__Point3D * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__Point3D__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__POINT3_D__STRUCT_H_
