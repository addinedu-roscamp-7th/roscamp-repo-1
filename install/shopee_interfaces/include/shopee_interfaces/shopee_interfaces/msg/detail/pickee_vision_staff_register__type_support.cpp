// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from shopee_interfaces:msg/PickeeVisionStaffRegister.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "shopee_interfaces/msg/detail/pickee_vision_staff_register__functions.h"
#include "shopee_interfaces/msg/detail/pickee_vision_staff_register__struct.hpp"
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

void PickeeVisionStaffRegister_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) shopee_interfaces::msg::PickeeVisionStaffRegister(_init);
}

void PickeeVisionStaffRegister_fini_function(void * message_memory)
{
  auto typed_message = static_cast<shopee_interfaces::msg::PickeeVisionStaffRegister *>(message_memory);
  typed_message->~PickeeVisionStaffRegister();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PickeeVisionStaffRegister_message_member_array[3] = {
  {
    "robot_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeVisionStaffRegister, robot_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "success",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeVisionStaffRegister, success),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "message",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeVisionStaffRegister, message),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PickeeVisionStaffRegister_message_members = {
  "shopee_interfaces::msg",  // message namespace
  "PickeeVisionStaffRegister",  // message name
  3,  // number of fields
  sizeof(shopee_interfaces::msg::PickeeVisionStaffRegister),
  false,  // has_any_key_member_
  PickeeVisionStaffRegister_message_member_array,  // message members
  PickeeVisionStaffRegister_init_function,  // function to initialize message memory (memory has to be allocated)
  PickeeVisionStaffRegister_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PickeeVisionStaffRegister_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PickeeVisionStaffRegister_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__PickeeVisionStaffRegister__get_type_hash,
  &shopee_interfaces__msg__PickeeVisionStaffRegister__get_type_description,
  &shopee_interfaces__msg__PickeeVisionStaffRegister__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace shopee_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<shopee_interfaces::msg::PickeeVisionStaffRegister>()
{
  return &::shopee_interfaces::msg::rosidl_typesupport_introspection_cpp::PickeeVisionStaffRegister_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, shopee_interfaces, msg, PickeeVisionStaffRegister)() {
  return &::shopee_interfaces::msg::rosidl_typesupport_introspection_cpp::PickeeVisionStaffRegister_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
