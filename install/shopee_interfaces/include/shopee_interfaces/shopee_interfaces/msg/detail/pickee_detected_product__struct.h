// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PickeeDetectedProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_detected_product.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'bbox_coords'
#include "shopee_interfaces/msg/detail/b_box__struct.h"

/// Struct defined in msg/PickeeDetectedProduct in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PickeeDetectedProduct
{
  int32_t product_id;
  int32_t bbox_number;
  shopee_interfaces__msg__BBox bbox_coords;
  float confidence;
} shopee_interfaces__msg__PickeeDetectedProduct;

// Struct for a sequence of shopee_interfaces__msg__PickeeDetectedProduct.
typedef struct shopee_interfaces__msg__PickeeDetectedProduct__Sequence
{
  shopee_interfaces__msg__PickeeDetectedProduct * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PickeeDetectedProduct__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__STRUCT_H_
