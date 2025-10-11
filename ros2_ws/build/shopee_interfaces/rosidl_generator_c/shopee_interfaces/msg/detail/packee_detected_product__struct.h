// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:msg/PackeeDetectedProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_detected_product.h"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__STRUCT_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'bbox'
#include "shopee_interfaces/msg/detail/b_box__struct.h"
// Member 'position'
#include "shopee_interfaces/msg/detail/point3_d__struct.h"

/// Struct defined in msg/PackeeDetectedProduct in the package shopee_interfaces.
typedef struct shopee_interfaces__msg__PackeeDetectedProduct
{
  int32_t product_id;
  shopee_interfaces__msg__BBox bbox;
  float confidence;
  shopee_interfaces__msg__Point3D position;
} shopee_interfaces__msg__PackeeDetectedProduct;

// Struct for a sequence of shopee_interfaces__msg__PackeeDetectedProduct.
typedef struct shopee_interfaces__msg__PackeeDetectedProduct__Sequence
{
  shopee_interfaces__msg__PackeeDetectedProduct * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__msg__PackeeDetectedProduct__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__STRUCT_H_
