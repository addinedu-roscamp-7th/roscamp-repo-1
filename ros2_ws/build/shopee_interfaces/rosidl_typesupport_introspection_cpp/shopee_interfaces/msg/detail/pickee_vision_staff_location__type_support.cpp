// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from shopee_interfaces:msg/PickeeVisionStaffLocation.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "shopee_interfaces/msg/detail/pickee_vision_staff_location__functions.h"
#include "shopee_interfaces/msg/detail/pickee_vision_staff_location__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace shopee_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void PickeeVisionStaffLocation_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) shopee_interfaces::msg::PickeeVisionStaffLocation(_init);
}

void PickeeVisionStaffLocation_fini_function(void * message_memory)
{
  auto typed_message = static_cast<shopee_interfaces::msg::PickeeVisionStaffLocation *>(message_memory);
  typed_message->~PickeeVisionStaffLocation();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PickeeVisionStaffLocation_message_member_array[4] = {
  {
    "robot_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeVisionStaffLocation, robot_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "relative_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<shopee_interfaces::msg::Point2D>(),  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeVisionStaffLocation, relative_position),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "distance",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeVisionStaffLocation, distance),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "is_tracking",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeVisionStaffLocation, is_tracking),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PickeeVisionStaffLocation_message_members = {
  "shopee_interfaces::msg",  // message namespace
  "PickeeVisionStaffLocation",  // message name
  4,  // number of fields
  sizeof(shopee_interfaces::msg::PickeeVisionStaffLocation),
  false,  // has_any_key_member_
  PickeeVisionStaffLocation_message_member_array,  // message members
  PickeeVisionStaffLocation_init_function,  // function to initialize message memory (memory has to be allocated)
  PickeeVisionStaffLocation_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PickeeVisionStaffLocation_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PickeeVisionStaffLocation_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__PickeeVisionStaffLocation__get_type_hash,
  &shopee_interfaces__msg__PickeeVisionStaffLocation__get_type_description,
  &shopee_interfaces__msg__PickeeVisionStaffLocation__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace shopee_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<shopee_interfaces::msg::PickeeVisionStaffLocation>()
{
  return &::shopee_interfaces::msg::rosidl_typesupport_introspection_cpp::PickeeVisionStaffLocation_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, shopee_interfaces, msg, PickeeVisionStaffLocation)() {
  return &::shopee_interfaces::msg::rosidl_typesupport_introspection_cpp::PickeeVisionStaffLocation_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
