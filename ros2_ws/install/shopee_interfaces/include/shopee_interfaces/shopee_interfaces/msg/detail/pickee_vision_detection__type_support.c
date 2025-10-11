// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from shopee_interfaces:msg/PickeeVisionDetection.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "shopee_interfaces/msg/detail/pickee_vision_detection__rosidl_typesupport_introspection_c.h"
#include "shopee_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "shopee_interfaces/msg/detail/pickee_vision_detection__functions.h"
#include "shopee_interfaces/msg/detail/pickee_vision_detection__struct.h"


// Include directives for member types
// Member `products`
#include "shopee_interfaces/msg/pickee_detected_product.h"
// Member `products`
#include "shopee_interfaces/msg/detail/pickee_detected_product__rosidl_typesupport_introspection_c.h"
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  shopee_interfaces__msg__PickeeVisionDetection__init(message_memory);
}

void shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_fini_function(void * message_memory)
{
  shopee_interfaces__msg__PickeeVisionDetection__fini(message_memory);
}

size_t shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__size_function__PickeeVisionDetection__products(
  const void * untyped_member)
{
  const shopee_interfaces__msg__PickeeDetectedProduct__Sequence * member =
    (const shopee_interfaces__msg__PickeeDetectedProduct__Sequence *)(untyped_member);
  return member->size;
}

const void * shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__get_const_function__PickeeVisionDetection__products(
  const void * untyped_member, size_t index)
{
  const shopee_interfaces__msg__PickeeDetectedProduct__Sequence * member =
    (const shopee_interfaces__msg__PickeeDetectedProduct__Sequence *)(untyped_member);
  return &member->data[index];
}

void * shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__get_function__PickeeVisionDetection__products(
  void * untyped_member, size_t index)
{
  shopee_interfaces__msg__PickeeDetectedProduct__Sequence * member =
    (shopee_interfaces__msg__PickeeDetectedProduct__Sequence *)(untyped_member);
  return &member->data[index];
}

void shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__fetch_function__PickeeVisionDetection__products(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const shopee_interfaces__msg__PickeeDetectedProduct * item =
    ((const shopee_interfaces__msg__PickeeDetectedProduct *)
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__get_const_function__PickeeVisionDetection__products(untyped_member, index));
  shopee_interfaces__msg__PickeeDetectedProduct * value =
    (shopee_interfaces__msg__PickeeDetectedProduct *)(untyped_value);
  *value = *item;
}

void shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__assign_function__PickeeVisionDetection__products(
  void * untyped_member, size_t index, const void * untyped_value)
{
  shopee_interfaces__msg__PickeeDetectedProduct * item =
    ((shopee_interfaces__msg__PickeeDetectedProduct *)
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__get_function__PickeeVisionDetection__products(untyped_member, index));
  const shopee_interfaces__msg__PickeeDetectedProduct * value =
    (const shopee_interfaces__msg__PickeeDetectedProduct *)(untyped_value);
  *item = *value;
}

bool shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__resize_function__PickeeVisionDetection__products(
  void * untyped_member, size_t size)
{
  shopee_interfaces__msg__PickeeDetectedProduct__Sequence * member =
    (shopee_interfaces__msg__PickeeDetectedProduct__Sequence *)(untyped_member);
  shopee_interfaces__msg__PickeeDetectedProduct__Sequence__fini(member);
  return shopee_interfaces__msg__PickeeDetectedProduct__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_member_array[5] = {
  {
    "robot_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PickeeVisionDetection, robot_id),  // bytes offset in struct
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
    offsetof(shopee_interfaces__msg__PickeeVisionDetection, order_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PickeeVisionDetection, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "products",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__msg__PickeeVisionDetection, products),  // bytes offset in struct
    NULL,  // default value
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__size_function__PickeeVisionDetection__products,  // size() function pointer
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__get_const_function__PickeeVisionDetection__products,  // get_const(index) function pointer
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__get_function__PickeeVisionDetection__products,  // get(index) function pointer
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__fetch_function__PickeeVisionDetection__products,  // fetch(index, &value) function pointer
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__assign_function__PickeeVisionDetection__products,  // assign(index, value) function pointer
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__resize_function__PickeeVisionDetection__products  // resize(index) function pointer
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
    offsetof(shopee_interfaces__msg__PickeeVisionDetection, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_members = {
  "shopee_interfaces__msg",  // message namespace
  "PickeeVisionDetection",  // message name
  5,  // number of fields
  sizeof(shopee_interfaces__msg__PickeeVisionDetection),
  false,  // has_any_key_member_
  shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_member_array,  // message members
  shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_init_function,  // function to initialize message memory (memory has to be allocated)
  shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_type_support_handle = {
  0,
  &shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__PickeeVisionDetection__get_type_hash,
  &shopee_interfaces__msg__PickeeVisionDetection__get_type_description,
  &shopee_interfaces__msg__PickeeVisionDetection__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_shopee_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, msg, PickeeVisionDetection)() {
  shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, msg, PickeeDetectedProduct)();
  if (!shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_type_support_handle.typesupport_identifier) {
    shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &shopee_interfaces__msg__PickeeVisionDetection__rosidl_typesupport_introspection_c__PickeeVisionDetection_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
