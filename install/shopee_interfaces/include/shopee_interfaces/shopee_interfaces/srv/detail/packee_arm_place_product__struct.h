// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:srv/PackeeArmPlaceProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/packee_arm_place_product.h"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PLACE_PRODUCT__STRUCT_H_
#define SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PLACE_PRODUCT__STRUCT_H_

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
// Member 'box_position'
#include "shopee_interfaces/msg/detail/point3_d__struct.h"

/// Struct defined in srv/PackeeArmPlaceProduct in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeArmPlaceProduct_Request
{
  int32_t robot_id;
  int32_t order_id;
  int32_t product_id;
  rosidl_runtime_c__String arm_side;
  shopee_interfaces__msg__Point3D box_position;
} shopee_interfaces__srv__PackeeArmPlaceProduct_Request;

// Struct for a sequence of shopee_interfaces__srv__PackeeArmPlaceProduct_Request.
typedef struct shopee_interfaces__srv__PackeeArmPlaceProduct_Request__Sequence
{
  shopee_interfaces__srv__PackeeArmPlaceProduct_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeArmPlaceProduct_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/PackeeArmPlaceProduct in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeArmPlaceProduct_Response
{
  bool accepted;
  rosidl_runtime_c__String message;
} shopee_interfaces__srv__PackeeArmPlaceProduct_Response;

// Struct for a sequence of shopee_interfaces__srv__PackeeArmPlaceProduct_Response.
typedef struct shopee_interfaces__srv__PackeeArmPlaceProduct_Response__Sequence
{
  shopee_interfaces__srv__PackeeArmPlaceProduct_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeArmPlaceProduct_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  shopee_interfaces__srv__PackeeArmPlaceProduct_Event__request__MAX_SIZE = 1
};
// response
enum
{
  shopee_interfaces__srv__PackeeArmPlaceProduct_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/PackeeArmPlaceProduct in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PackeeArmPlaceProduct_Event
{
  service_msgs__msg__ServiceEventInfo info;
  shopee_interfaces__srv__PackeeArmPlaceProduct_Request__Sequence request;
  shopee_interfaces__srv__PackeeArmPlaceProduct_Response__Sequence response;
} shopee_interfaces__srv__PackeeArmPlaceProduct_Event;

// Struct for a sequence of shopee_interfaces__srv__PackeeArmPlaceProduct_Event.
typedef struct shopee_interfaces__srv__PackeeArmPlaceProduct_Event__Sequence
{
  shopee_interfaces__srv__PackeeArmPlaceProduct_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PackeeArmPlaceProduct_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PLACE_PRODUCT__STRUCT_H_
