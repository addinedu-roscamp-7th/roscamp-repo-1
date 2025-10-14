// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:srv/MainGetAvailableRobots.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/main_get_available_robots.h"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_AVAILABLE_ROBOTS__STRUCT_H_
#define SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_AVAILABLE_ROBOTS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'robot_type'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/MainGetAvailableRobots in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__MainGetAvailableRobots_Request
{
  /// "pickee" or "packee" or "" (empty for all)
  rosidl_runtime_c__String robot_type;
} shopee_interfaces__srv__MainGetAvailableRobots_Request;

// Struct for a sequence of shopee_interfaces__srv__MainGetAvailableRobots_Request.
typedef struct shopee_interfaces__srv__MainGetAvailableRobots_Request__Sequence
{
  shopee_interfaces__srv__MainGetAvailableRobots_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__MainGetAvailableRobots_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'robot_ids'
#include "rosidl_runtime_c/primitives_sequence.h"
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/MainGetAvailableRobots in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__MainGetAvailableRobots_Response
{
  rosidl_runtime_c__int32__Sequence robot_ids;
  bool success;
  rosidl_runtime_c__String message;
} shopee_interfaces__srv__MainGetAvailableRobots_Response;

// Struct for a sequence of shopee_interfaces__srv__MainGetAvailableRobots_Response.
typedef struct shopee_interfaces__srv__MainGetAvailableRobots_Response__Sequence
{
  shopee_interfaces__srv__MainGetAvailableRobots_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__MainGetAvailableRobots_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  shopee_interfaces__srv__MainGetAvailableRobots_Event__request__MAX_SIZE = 1
};
// response
enum
{
  shopee_interfaces__srv__MainGetAvailableRobots_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/MainGetAvailableRobots in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__MainGetAvailableRobots_Event
{
  service_msgs__msg__ServiceEventInfo info;
  shopee_interfaces__srv__MainGetAvailableRobots_Request__Sequence request;
  shopee_interfaces__srv__MainGetAvailableRobots_Response__Sequence response;
} shopee_interfaces__srv__MainGetAvailableRobots_Event;

// Struct for a sequence of shopee_interfaces__srv__MainGetAvailableRobots_Event.
typedef struct shopee_interfaces__srv__MainGetAvailableRobots_Event__Sequence
{
  shopee_interfaces__srv__MainGetAvailableRobots_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__MainGetAvailableRobots_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_AVAILABLE_ROBOTS__STRUCT_H_
