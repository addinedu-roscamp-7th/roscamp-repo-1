// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from shopee_interfaces:srv/MainGetLocationPose.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/main_get_location_pose.h"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_LOCATION_POSE__STRUCT_H_
#define SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_LOCATION_POSE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/MainGetLocationPose in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__MainGetLocationPose_Request
{
  int32_t location_id;
} shopee_interfaces__srv__MainGetLocationPose_Request;

// Struct for a sequence of shopee_interfaces__srv__MainGetLocationPose_Request.
typedef struct shopee_interfaces__srv__MainGetLocationPose_Request__Sequence
{
  shopee_interfaces__srv__MainGetLocationPose_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__MainGetLocationPose_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'pose'
#include "shopee_interfaces/msg/detail/pose2_d__struct.h"
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/MainGetLocationPose in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__MainGetLocationPose_Response
{
  shopee_interfaces__msg__Pose2D pose;
  bool success;
  rosidl_runtime_c__String message;
} shopee_interfaces__srv__MainGetLocationPose_Response;

// Struct for a sequence of shopee_interfaces__srv__MainGetLocationPose_Response.
typedef struct shopee_interfaces__srv__MainGetLocationPose_Response__Sequence
{
  shopee_interfaces__srv__MainGetLocationPose_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__MainGetLocationPose_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  shopee_interfaces__srv__MainGetLocationPose_Event__request__MAX_SIZE = 1
};
// response
enum
{
  shopee_interfaces__srv__MainGetLocationPose_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/MainGetLocationPose in the package shopee_interfaces.
typedef struct shopee_interfaces__srv__MainGetLocationPose_Event
{
  service_msgs__msg__ServiceEventInfo info;
  shopee_interfaces__srv__MainGetLocationPose_Request__Sequence request;
  shopee_interfaces__srv__MainGetLocationPose_Response__Sequence response;
} shopee_interfaces__srv__MainGetLocationPose_Event;

// Struct for a sequence of shopee_interfaces__srv__MainGetLocationPose_Event.
typedef struct shopee_interfaces__srv__MainGetLocationPose_Event__Sequence
{
  shopee_interfaces__srv__MainGetLocationPose_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} shopee_interfaces__srv__MainGetLocationPose_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_LOCATION_POSE__STRUCT_H_
