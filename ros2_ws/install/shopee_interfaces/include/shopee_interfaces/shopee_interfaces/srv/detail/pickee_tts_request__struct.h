// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:srv/PickeeTtsRequest.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_tts_request.h"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_TTS_REQUEST__STRUCT_H_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_TTS_REQUEST__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'text_to_speak'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/PickeeTtsRequest in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeTtsRequest_Request
{
  rosidl_runtime_c__String text_to_speak;
} shopee_interfaces__srv__PickeeTtsRequest_Request;

// Struct for a sequence of shopee_interfaces__srv__PickeeTtsRequest_Request.
typedef struct shopee_interfaces__srv__PickeeTtsRequest_Request__Sequence
{
  shopee_interfaces__srv__PickeeTtsRequest_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeTtsRequest_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/PickeeTtsRequest in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeTtsRequest_Response
{
  bool success;
  rosidl_runtime_c__String message;
} shopee_interfaces__srv__PickeeTtsRequest_Response;

// Struct for a sequence of shopee_interfaces__srv__PickeeTtsRequest_Response.
typedef struct shopee_interfaces__srv__PickeeTtsRequest_Response__Sequence
{
  shopee_interfaces__srv__PickeeTtsRequest_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeTtsRequest_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  shopee_interfaces__srv__PickeeTtsRequest_Event__request__MAX_SIZE = 1
};
// response
enum
{
  shopee_interfaces__srv__PickeeTtsRequest_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/PickeeTtsRequest in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__PickeeTtsRequest_Event
{
  service_msgs__msg__ServiceEventInfo info;
  shopee_interfaces__srv__PickeeTtsRequest_Request__Sequence request;
  shopee_interfaces__srv__PickeeTtsRequest_Response__Sequence response;
} shopee_interfaces__srv__PickeeTtsRequest_Event;

// Struct for a sequence of shopee_interfaces__srv__PickeeTtsRequest_Event.
typedef struct shopee_interfaces__srv__PickeeTtsRequest_Event__Sequence
{
  shopee_interfaces__srv__PickeeTtsRequest_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__PickeeTtsRequest_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_TTS_REQUEST__STRUCT_H_
