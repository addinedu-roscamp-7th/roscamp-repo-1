// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/Obstacle.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/obstacle.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'obstacle_type'
#include "rosidl_runtime_c/string.h"
// Member 'position'
#include "shopee_interfaces/msg/detail/point2_d__struct.h"
// Member 'direction'
#include "shopee_interfaces/msg/detail/vector2_d__struct.h"
// Member 'bbox'
#include "shopee_interfaces/msg/detail/b_box__struct.h"

/// Struct defined in msg/Obstacle in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__Obstacle
{
  rosidl_runtime_c__String obstacle_type;
  shopee_interfaces__msg__Point2D position;
  float distance;
  float velocity;
  shopee_interfaces__msg__Vector2D direction;
  shopee_interfaces__msg__BBox bbox;
  float confidence;
} shopee_interfaces__msg__Obstacle;

// Struct for a sequence of shopee_interfaces__msg__Obstacle.
typedef struct shopee_interfaces__msg__Obstacle__Sequence
{
  shopee_interfaces__msg__Obstacle * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__Obstacle__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__STRUCT_H_
