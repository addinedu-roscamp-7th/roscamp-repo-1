// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:srv/PackeeVisionDetectProductsInCart.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/packee_vision_detect_products_in_cart.h"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_DETECT_PRODUCTS_IN_CART__STRUCT_H_
#define SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_DETECT_PRODUCTS_IN_CART__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'expected_product_ids'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/PackeeVisionDetectProductsInCart in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request
{
  int32_t robot_id;
  int32_t order_id;
  rosidl_runtime_c__int32__Sequence expected_product_ids;
} shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request;

// Struct for a sequence of shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request.
typedef struct shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'products'
#include "shopee_interfaces/msg/detail/packee_detected_product__struct.h"
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/PackeeVisionDetectProductsInCart in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response
{
  bool success;
  shopee_interfaces__msg__PackeeDetectedProduct__Sequence products;
  int32_t total_detected;
  rosidl_runtime_c__String message;
} shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response;

// Struct for a sequence of shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response.
typedef struct shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__request__MAX_SIZE = 1
};
// response
enum
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/PackeeVisionDetectProductsInCart in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event
{
  service_msgs__msg__ServiceEventInfo info;
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence request;
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence response;
} shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event;

// Struct for a sequence of shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event.
typedef struct shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__Sequence
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_DETECT_PRODUCTS_IN_CART__STRUCT_H_
