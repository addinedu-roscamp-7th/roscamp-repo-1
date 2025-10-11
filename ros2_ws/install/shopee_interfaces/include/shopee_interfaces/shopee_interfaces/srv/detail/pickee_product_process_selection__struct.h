// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:srv/PickeeProductProcessSelection.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_product_process_selection.h"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_PRODUCT_PROCESS_SELECTION__STRUCT_H_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_PRODUCT_PROCESS_SELECTION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/PickeeProductProcessSelection in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeProductProcessSelection_Request
{
  int32_t robot_id;
  int32_t order_id;
  int32_t product_id;
  int32_t bbox_number;
} shopee_interfaces__srv__PickeeProductProcessSelection_Request;

// Struct for a sequence of shopee_interfaces__srv__PickeeProductProcessSelection_Request.
typedef struct shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence
{
  shopee_interfaces__srv__PickeeProductProcessSelection_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/PickeeProductProcessSelection in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeProductProcessSelection_Response
{
  bool success;
  rosidl_runtime_c__String message;
} shopee_interfaces__srv__PickeeProductProcessSelection_Response;

// Struct for a sequence of shopee_interfaces__srv__PickeeProductProcessSelection_Response.
typedef struct shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence
{
  shopee_interfaces__srv__PickeeProductProcessSelection_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  shopee_interfaces__srv__PickeeProductProcessSelection_Event__request__MAX_SIZE = 1
};
// response
enum
{
  shopee_interfaces__srv__PickeeProductProcessSelection_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/PickeeProductProcessSelection in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeProductProcessSelection_Event
{
  service_msgs__msg__ServiceEventInfo info;
  shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence request;
  shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence response;
} shopee_interfaces__srv__PickeeProductProcessSelection_Event;

// Struct for a sequence of shopee_interfaces__srv__PickeeProductProcessSelection_Event.
typedef struct shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence
{
  shopee_interfaces__srv__PickeeProductProcessSelection_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_PRODUCT_PROCESS_SELECTION__STRUCT_H_
