// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/ProductLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/product_location.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/ProductLocation in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__ProductLocation
{
  int32_t product_id;
  int32_t location_id;
  int32_t section_id;
  int32_t quantity;
} shopee_interfaces__msg__ProductLocation;

// Struct for a sequence of shopee_interfaces__msg__ProductLocation.
typedef struct shopee_interfaces__msg__ProductLocation__Sequence
{
  shopee_interfaces__msg__ProductLocation * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__ProductLocation__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__STRUCT_H_
