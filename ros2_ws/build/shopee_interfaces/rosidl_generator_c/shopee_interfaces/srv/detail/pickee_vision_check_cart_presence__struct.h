// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:srv/PickeeVisionCheckCartPresence.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_vision_check_cart_presence.h"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_VISION_CHECK_CART_PRESENCE__STRUCT_H_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_VISION_CHECK_CART_PRESENCE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/PickeeVisionCheckCartPresence in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeVisionCheckCartPresence_Request
{
  int32_t robot_id;
  int32_t order_id;
} shopee_interfaces__srv__PickeeVisionCheckCartPresence_Request;

// Struct for a sequence of shopee_interfaces__srv__PickeeVisionCheckCartPresence_Request.
typedef struct shopee_interfaces__srv__PickeeVisionCheckCartPresence_Request__Sequence
{
  shopee_interfaces__srv__PickeeVisionCheckCartPresence_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeVisionCheckCartPresence_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/PickeeVisionCheckCartPresence in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeVisionCheckCartPresence_Response
{
  bool success;
  bool cart_present;
  rosidl_runtime_c__String message;
} shopee_interfaces__srv__PickeeVisionCheckCartPresence_Response;

// Struct for a sequence of shopee_interfaces__srv__PickeeVisionCheckCartPresence_Response.
typedef struct shopee_interfaces__srv__PickeeVisionCheckCartPresence_Response__Sequence
{
  shopee_interfaces__srv__PickeeVisionCheckCartPresence_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeVisionCheckCartPresence_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  shopee_interfaces__srv__PickeeVisionCheckCartPresence_Event__request__MAX_SIZE = 1
};
// response
enum
{
  shopee_interfaces__srv__PickeeVisionCheckCartPresence_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/PickeeVisionCheckCartPresence in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeVisionCheckCartPresence_Event
{
  service_msgs__msg__ServiceEventInfo info;
  shopee_interfaces__srv__PickeeVisionCheckCartPresence_Request__Sequence request;
  shopee_interfaces__srv__PickeeVisionCheckCartPresence_Response__Sequence response;
} shopee_interfaces__srv__PickeeVisionCheckCartPresence_Event;

// Struct for a sequence of shopee_interfaces__srv__PickeeVisionCheckCartPresence_Event.
typedef struct shopee_interfaces__srv__PickeeVisionCheckCartPresence_Event__Sequence
{
  shopee_interfaces__srv__PickeeVisionCheckCartPresence_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeVisionCheckCartPresence_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_VISION_CHECK_CART_PRESENCE__STRUCT_H_
