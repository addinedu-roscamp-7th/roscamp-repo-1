// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from shopee_interfaces:msg/Obstacle.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "shopee_interfaces/msg/detail/obstacle__rosidl_typesupport_introspection_c.h"
#include "shopee_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "shopee_interfaces/msg/detail/obstacle__functions.h"
#include "shopee_interfaces/msg/detail/obstacle__struct.h"


// Include directives for member types
// Member `obstacle_type`
#include "rosidl_runtime_c/string_functions.h"
// Member `position`
#include "shopee_interfaces/msg/point2_d.h"
// Member `position`
#include "shopee_interfaces/msg/detail/point2_d__rosidl_typesupport_introspection_c.h"
// Member `direction`
#include "shopee_interfaces/msg/vector2_d.h"
// Member `direction`
#include "shopee_interfaces/msg/detail/vector2_d__rosidl_typesupport_introspection_c.h"
// Member `bbox`
#include "shopee_interfaces/msg/b_box.h"
// Member `bbox`
#include "shopee_interfaces/msg/detail/b_box__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  shopee_interfaces__msg__Obstacle__init(message_memory);
}

void shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_fini_function(void * message_memory)
{
  shopee_interfaces__msg__Obstacle__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_member_array[7] = {
  {
    "obstacle_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__Obstacle, obstacle_type),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__Obstacle, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "distance",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__Obstacle, distance),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__Obstacle, velocity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "direction",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__Obstacle, direction),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "bbox",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__Obstacle, bbox),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "confidence",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__Obstacle, confidence),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_members = {
  "shopee_interfaces__msg",  // message namespace
  "Obstacle",  // message name
  7,  // number of fields
  sizeof(shopee_interfaces__msg__Obstacle),
  false,  // has_any_key_member_
  shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_member_array,  // message members
  shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_init_function,  // function to initialize message memory (memory has to be allocated)
  shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_type_support_handle = {
  0,
  &shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__Obstacle__get_type_hash,
  &shopee_interfaces__msg__Obstacle__get_type_description,
  &shopee_interfaces__msg__Obstacle__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_shopee_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, msg, Obstacle)() {
  shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, msg, Point2D)();
  shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, msg, Vector2D)();
  shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_member_array[5].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, msg, BBox)();
  if (!shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_type_support_handle.typesupport_identifier) {
    shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &shopee_interfaces__msg__Obstacle__rosidl_typesupport_introspection_c__Obstacle_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
