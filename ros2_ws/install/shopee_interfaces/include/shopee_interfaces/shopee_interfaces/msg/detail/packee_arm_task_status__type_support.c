// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from shopee_interfaces:msg/PackeeArmTaskStatus.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "shopee_interfaces/msg/detail/packee_arm_task_status__rosidl_typesupport_introspection_c.h"
#include "shopee_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "shopee_interfaces/msg/detail/packee_arm_task_status__functions.h"
#include "shopee_interfaces/msg/detail/packee_arm_task_status__struct.h"


// Include directives for member types
// Member `arm_side`
// Member `status`
// Member `current_phase`
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  shopee_interfaces__msg__PackeeArmTaskStatus__init(message_memory);
}

void shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_fini_function(void * message_memory)
{
  shopee_interfaces__msg__PackeeArmTaskStatus__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_message_member_array[8] = {
  {
    "robot_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PackeeArmTaskStatus, robot_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "order_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PackeeArmTaskStatus, order_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "product_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PackeeArmTaskStatus, product_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "arm_side",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PackeeArmTaskStatus, arm_side),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PackeeArmTaskStatus, status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "current_phase",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PackeeArmTaskStatus, current_phase),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "progress",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PackeeArmTaskStatus, progress),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PackeeArmTaskStatus, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_message_members = {
  "shopee_interfaces__msg",  // message namespace
  "PackeeArmTaskStatus",  // message name
  8,  // number of fields
  sizeof(shopee_interfaces__msg__PackeeArmTaskStatus),
  false,  // has_any_key_member_
  shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_message_member_array,  // message members
  shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_message_type_support_handle = {
  0,
  &shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__PackeeArmTaskStatus__get_type_hash,
  &shopee_interfaces__msg__PackeeArmTaskStatus__get_type_description,
  &shopee_interfaces__msg__PackeeArmTaskStatus__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_shopee_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, msg, PackeeArmTaskStatus)() {
  if (!shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_message_type_support_handle.typesupport_identifier) {
    shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &shopee_interfaces__msg__PackeeArmTaskStatus__rosidl_typesupport_introspection_c__PackeeArmTaskStatus_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
