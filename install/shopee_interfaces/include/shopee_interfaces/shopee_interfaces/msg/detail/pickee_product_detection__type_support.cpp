// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from shopee_interfaces:msg/PickeeProductDetection.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "shopee_interfaces/msg/detail/pickee_product_detection__functions.h"
#include "shopee_interfaces/msg/detail/pickee_product_detection__struct.hpp"
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

void PickeeProductDetection_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) shopee_interfaces::msg::PickeeProductDetection(_init);
}

void PickeeProductDetection_fini_function(void * message_memory)
{
  auto typed_message = static_cast<shopee_interfaces::msg::PickeeProductDetection *>(message_memory);
  typed_message->~PickeeProductDetection();
}

size_t size_function__PickeeProductDetection__products(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<shopee_interfaces::msg::PickeeDetectedProduct> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PickeeProductDetection__products(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<shopee_interfaces::msg::PickeeDetectedProduct> *>(untyped_member);
  return &member[index];
}

void * get_function__PickeeProductDetection__products(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<shopee_interfaces::msg::PickeeDetectedProduct> *>(untyped_member);
  return &member[index];
}

void fetch_function__PickeeProductDetection__products(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const shopee_interfaces::msg::PickeeDetectedProduct *>(
    get_const_function__PickeeProductDetection__products(untyped_member, index));
  auto & value = *reinterpret_cast<shopee_interfaces::msg::PickeeDetectedProduct *>(untyped_value);
  value = item;
}

void assign_function__PickeeProductDetection__products(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<shopee_interfaces::msg::PickeeDetectedProduct *>(
    get_function__PickeeProductDetection__products(untyped_member, index));
  const auto & value = *reinterpret_cast<const shopee_interfaces::msg::PickeeDetectedProduct *>(untyped_value);
  item = value;
}

void resize_function__PickeeProductDetection__products(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<shopee_interfaces::msg::PickeeDetectedProduct> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PickeeProductDetection_message_member_array[3] = {
  {
    "robot_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeProductDetection, robot_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "order_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeProductDetection, order_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "products",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<shopee_interfaces::msg::PickeeDetectedProduct>(),  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces::msg::PickeeProductDetection, products),  // bytes offset in struct
    nullptr,  // default value
    size_function__PickeeProductDetection__products,  // size() function pointer
    get_const_function__PickeeProductDetection__products,  // get_const(index) function pointer
    get_function__PickeeProductDetection__products,  // get(index) function pointer
    fetch_function__PickeeProductDetection__products,  // fetch(index, &value) function pointer
    assign_function__PickeeProductDetection__products,  // assign(index, value) function pointer
    resize_function__PickeeProductDetection__products  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PickeeProductDetection_message_members = {
  "shopee_interfaces::msg",  // message namespace
  "PickeeProductDetection",  // message name
  3,  // number of fields
  sizeof(shopee_interfaces::msg::PickeeProductDetection),
  false,  // has_any_key_member_
  PickeeProductDetection_message_member_array,  // message members
  PickeeProductDetection_init_function,  // function to initialize message memory (memory has to be allocated)
  PickeeProductDetection_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PickeeProductDetection_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PickeeProductDetection_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__PickeeProductDetection__get_type_hash,
  &shopee_interfaces__msg__PickeeProductDetection__get_type_description,
  &shopee_interfaces__msg__PickeeProductDetection__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace shopee_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<shopee_interfaces::msg::PickeeProductDetection>()
{
  return &::shopee_interfaces::msg::rosidl_typesupport_introspection_cpp::PickeeProductDetection_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, shopee_interfaces, msg, PickeeProductDetection)() {
  return &::shopee_interfaces::msg::rosidl_typesupport_introspection_cpp::PickeeProductDetection_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
