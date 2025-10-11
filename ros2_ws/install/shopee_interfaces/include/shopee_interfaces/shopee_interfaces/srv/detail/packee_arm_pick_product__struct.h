// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:srv/PackeeArmPickProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/packee_arm_pick_product.h"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PICK_PRODUCT__STRUCT_H_
#define SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PICK_PRODUCT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'arm_side'
#include "rosidl_runtime_c/string.h"
// Member 'target_position'
#include "shopee_interfaces/msg/detail/point3_d__struct.h"
// Member 'bbox'
#include "shopee_interfaces/msg/detail/b_box__struct.h"

/// Struct defined in srv/PackeeArmPickProduct in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeArmPickProduct_Request
{
  int32_t robot_id;
  int32_t order_id;
  int32_t product_id;
  rosidl_runtime_c__String arm_side;
  shopee_interfaces__msg__Point3D target_position;
  shopee_interfaces__msg__BBox bbox;
} shopee_interfaces__srv__PackeeArmPickProduct_Request;

// Struct for a sequence of shopee_interfaces__srv__PackeeArmPickProduct_Request.
typedef struct shopee_interfaces__srv__PackeeArmPickProduct_Request__Sequence
{
  shopee_interfaces__srv__PackeeArmPickProduct_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeArmPickProduct_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/PackeeArmPickProduct in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeArmPickProduct_Response
{
  bool accepted;
  rosidl_runtime_c__String message;
} shopee_interfaces__srv__PackeeArmPickProduct_Response;

// Struct for a sequence of shopee_interfaces__srv__PackeeArmPickProduct_Response.
typedef struct shopee_interfaces__srv__PackeeArmPickProduct_Response__Sequence
{
  shopee_interfaces__srv__PackeeArmPickProduct_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeArmPickProduct_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  shopee_interfaces__srv__PackeeArmPickProduct_Event__request__MAX_SIZE = 1
};
// response
enum
{
  shopee_interfaces__srv__PackeeArmPickProduct_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/PackeeArmPickProduct in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeArmPickProduct_Event
{
  service_msgs__msg__ServiceEventInfo info;
  shopee_interfaces__srv__PackeeArmPickProduct_Request__Sequence request;
  shopee_interfaces__srv__PackeeArmPickProduct_Response__Sequence response;
} shopee_interfaces__srv__PackeeArmPickProduct_Event;

// Struct for a sequence of shopee_interfaces__srv__PackeeArmPickProduct_Event.
typedef struct shopee_interfaces__srv__PackeeArmPickProduct_Event__Sequence
{
  shopee_interfaces__srv__PackeeArmPickProduct_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeArmPickProduct_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PICK_PRODUCT__STRUCT_H_
