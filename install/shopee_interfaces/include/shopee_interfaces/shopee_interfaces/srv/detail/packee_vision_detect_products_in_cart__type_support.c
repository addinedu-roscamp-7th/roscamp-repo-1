// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from shopee_interfaces:srv/PackeeVisionDetectProductsInCart.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c.h"
#include "shopee_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__functions.h"
#include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__struct.h"


// Include directives for member types
// Member `expected_product_ids`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__init(message_memory);
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_fini_function(void * message_memory)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__fini(message_memory);
}

size_t shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__size_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids(
  const void * untyped_member)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return member->size;
}

const void * shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__fetch_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__assign_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

bool shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__resize_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  rosidl_runtime_c__int32__Sequence__fini(member);
  return rosidl_runtime_c__int32__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_member_array[3] = {
  {
    "robot_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request, robot_id),  // bytes offset in struct
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
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request, order_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "expected_product_ids",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request, expected_product_ids),  // bytes offset in struct
    NULL,  // default value
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__size_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids,  // size() function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids,  // get_const(index) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids,  // get(index) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__fetch_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids,  // fetch(index, &value) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__assign_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids,  // assign(index, value) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__resize_function__PackeeVisionDetectProductsInCart_Request__expected_product_ids  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_members = {
  "shopee_interfaces__srv",  // message namespace
  "PackeeVisionDetectProductsInCart_Request",  // message name
  3,  // number of fields
  sizeof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request),
  false,  // has_any_key_member_
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_member_array,  // message members
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_type_support_handle = {
  0,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__get_type_hash,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__get_type_description,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_shopee_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Request)() {
  if (!shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_type_support_handle.typesupport_identifier) {
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c.h"
// already included above
// #include "shopee_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__functions.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__struct.h"


// Include directives for member types
// Member `products`
#include "shopee_interfaces/msg/packee_detected_product.h"
// Member `products`
#include "shopee_interfaces/msg/detail/packee_detected_product__rosidl_typesupport_introspection_c.h"
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__init(message_memory);
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_fini_function(void * message_memory)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__fini(message_memory);
}

size_t shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__size_function__PackeeVisionDetectProductsInCart_Response__products(
  const void * untyped_member)
{
  const shopee_interfaces__msg__PackeeDetectedProduct__Sequence * member =
    (const shopee_interfaces__msg__PackeeDetectedProduct__Sequence *)(untyped_member);
  return member->size;
}

const void * shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Response__products(
  const void * untyped_member, size_t index)
{
  const shopee_interfaces__msg__PackeeDetectedProduct__Sequence * member =
    (const shopee_interfaces__msg__PackeeDetectedProduct__Sequence *)(untyped_member);
  return &member->data[index];
}

void * shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Response__products(
  void * untyped_member, size_t index)
{
  shopee_interfaces__msg__PackeeDetectedProduct__Sequence * member =
    (shopee_interfaces__msg__PackeeDetectedProduct__Sequence *)(untyped_member);
  return &member->data[index];
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__fetch_function__PackeeVisionDetectProductsInCart_Response__products(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const shopee_interfaces__msg__PackeeDetectedProduct * item =
    ((const shopee_interfaces__msg__PackeeDetectedProduct *)
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Response__products(untyped_member, index));
  shopee_interfaces__msg__PackeeDetectedProduct * value =
    (shopee_interfaces__msg__PackeeDetectedProduct *)(untyped_value);
  *value = *item;
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__assign_function__PackeeVisionDetectProductsInCart_Response__products(
  void * untyped_member, size_t index, const void * untyped_value)
{
  shopee_interfaces__msg__PackeeDetectedProduct * item =
    ((shopee_interfaces__msg__PackeeDetectedProduct *)
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Response__products(untyped_member, index));
  const shopee_interfaces__msg__PackeeDetectedProduct * value =
    (const shopee_interfaces__msg__PackeeDetectedProduct *)(untyped_value);
  *item = *value;
}

bool shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__resize_function__PackeeVisionDetectProductsInCart_Response__products(
  void * untyped_member, size_t size)
{
  shopee_interfaces__msg__PackeeDetectedProduct__Sequence * member =
    (shopee_interfaces__msg__PackeeDetectedProduct__Sequence *)(untyped_member);
  shopee_interfaces__msg__PackeeDetectedProduct__Sequence__fini(member);
  return shopee_interfaces__msg__PackeeDetectedProduct__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_member_array[4] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response, success),  // bytes offset in struct
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
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response, products),  // bytes offset in struct
    NULL,  // default value
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__size_function__PackeeVisionDetectProductsInCart_Response__products,  // size() function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Response__products,  // get_const(index) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Response__products,  // get(index) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__fetch_function__PackeeVisionDetectProductsInCart_Response__products,  // fetch(index, &value) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__assign_function__PackeeVisionDetectProductsInCart_Response__products,  // assign(index, value) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__resize_function__PackeeVisionDetectProductsInCart_Response__products  // resize(index) function pointer
  },
  {
    "total_detected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response, total_detected),  // bytes offset in struct
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
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_members = {
  "shopee_interfaces__srv",  // message namespace
  "PackeeVisionDetectProductsInCart_Response",  // message name
  4,  // number of fields
  sizeof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response),
  false,  // has_any_key_member_
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_member_array,  // message members
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_type_support_handle = {
  0,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__get_type_hash,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__get_type_description,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_shopee_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Response)() {
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, msg, PackeeDetectedProduct)();
  if (!shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_type_support_handle.typesupport_identifier) {
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c.h"
// already included above
// #include "shopee_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__functions.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__struct.h"


// Include directives for member types
// Member `info`
#include "service_msgs/msg/service_event_info.h"
// Member `info`
#include "service_msgs/msg/detail/service_event_info__rosidl_typesupport_introspection_c.h"
// Member `request`
// Member `response`
#include "shopee_interfaces/srv/packee_vision_detect_products_in_cart.h"
// Member `request`
// Member `response`
// already included above
// #include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__init(message_memory);
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_fini_function(void * message_memory)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__fini(message_memory);
}

size_t shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__size_function__PackeeVisionDetectProductsInCart_Event__request(
  const void * untyped_member)
{
  const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence * member =
    (const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence *)(untyped_member);
  return member->size;
}

const void * shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Event__request(
  const void * untyped_member, size_t index)
{
  const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence * member =
    (const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void * shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Event__request(
  void * untyped_member, size_t index)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence * member =
    (shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__fetch_function__PackeeVisionDetectProductsInCart_Event__request(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request * item =
    ((const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request *)
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Event__request(untyped_member, index));
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request * value =
    (shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request *)(untyped_value);
  *value = *item;
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__assign_function__PackeeVisionDetectProductsInCart_Event__request(
  void * untyped_member, size_t index, const void * untyped_value)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request * item =
    ((shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request *)
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Event__request(untyped_member, index));
  const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request * value =
    (const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request *)(untyped_value);
  *item = *value;
}

bool shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__resize_function__PackeeVisionDetectProductsInCart_Event__request(
  void * untyped_member, size_t size)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence * member =
    (shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence *)(untyped_member);
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence__fini(member);
  return shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__Sequence__init(member, size);
}

size_t shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__size_function__PackeeVisionDetectProductsInCart_Event__response(
  const void * untyped_member)
{
  const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence * member =
    (const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence *)(untyped_member);
  return member->size;
}

const void * shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Event__response(
  const void * untyped_member, size_t index)
{
  const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence * member =
    (const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void * shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Event__response(
  void * untyped_member, size_t index)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence * member =
    (shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__fetch_function__PackeeVisionDetectProductsInCart_Event__response(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response * item =
    ((const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response *)
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Event__response(untyped_member, index));
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response * value =
    (shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response *)(untyped_value);
  *value = *item;
}

void shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__assign_function__PackeeVisionDetectProductsInCart_Event__response(
  void * untyped_member, size_t index, const void * untyped_value)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response * item =
    ((shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response *)
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Event__response(untyped_member, index));
  const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response * value =
    (const shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response *)(untyped_value);
  *item = *value;
}

bool shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__resize_function__PackeeVisionDetectProductsInCart_Event__response(
  void * untyped_member, size_t size)
{
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence * member =
    (shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence *)(untyped_member);
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence__fini(member);
  return shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_member_array[3] = {
  {
    "info",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event, info),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "request",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event, request),  // bytes offset in struct
    NULL,  // default value
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__size_function__PackeeVisionDetectProductsInCart_Event__request,  // size() function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Event__request,  // get_const(index) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Event__request,  // get(index) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__fetch_function__PackeeVisionDetectProductsInCart_Event__request,  // fetch(index, &value) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__assign_function__PackeeVisionDetectProductsInCart_Event__request,  // assign(index, value) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__resize_function__PackeeVisionDetectProductsInCart_Event__request  // resize(index) function pointer
  },
  {
    "response",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event, response),  // bytes offset in struct
    NULL,  // default value
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__size_function__PackeeVisionDetectProductsInCart_Event__response,  // size() function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_const_function__PackeeVisionDetectProductsInCart_Event__response,  // get_const(index) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__get_function__PackeeVisionDetectProductsInCart_Event__response,  // get(index) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__fetch_function__PackeeVisionDetectProductsInCart_Event__response,  // fetch(index, &value) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__assign_function__PackeeVisionDetectProductsInCart_Event__response,  // assign(index, value) function pointer
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__resize_function__PackeeVisionDetectProductsInCart_Event__response  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_members = {
  "shopee_interfaces__srv",  // message namespace
  "PackeeVisionDetectProductsInCart_Event",  // message name
  3,  // number of fields
  sizeof(shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event),
  false,  // has_any_key_member_
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_member_array,  // message members
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_init_function,  // function to initialize message memory (memory has to be allocated)
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_type_support_handle = {
  0,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_members,
  get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__get_type_hash,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__get_type_description,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_shopee_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Event)() {
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, service_msgs, msg, ServiceEventInfo)();
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Request)();
  shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Response)();
  if (!shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_type_support_handle.typesupport_identifier) {
    shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "shopee_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_service_members = {
  "shopee_interfaces__srv",  // service namespace
  "PackeeVisionDetectProductsInCart",  // service name
  // the following fields are initialized below on first access
  NULL,  // request message
  // shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_type_support_handle,
  NULL,  // response message
  // shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_type_support_handle
  NULL  // event_message
  // shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_type_support_handle
};


static rosidl_service_type_support_t shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_service_type_support_handle = {
  0,
  &shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_service_members,
  get_service_typesupport_handle_function,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Request__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Request_message_type_support_handle,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Response__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Response_message_type_support_handle,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart_Event__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    shopee_interfaces,
    srv,
    PackeeVisionDetectProductsInCart
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    shopee_interfaces,
    srv,
    PackeeVisionDetectProductsInCart
  ),
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart__get_type_hash,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart__get_type_description,
  &shopee_interfaces__srv__PackeeVisionDetectProductsInCart__get_type_description_sources,
};

// Forward declaration of message type support functions for service members
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Request)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Response)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Event)(void);

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_shopee_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart)(void) {
  if (!shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_service_type_support_handle.typesupport_identifier) {
    shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Response)()->data;
  }
  if (!service_members->event_members_) {
    service_members->event_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shopee_interfaces, srv, PackeeVisionDetectProductsInCart_Event)()->data;
  }

  return &shopee_interfaces__srv__detail__packee_vision_detect_products_in_cart__rosidl_typesupport_introspection_c__PackeeVisionDetectProductsInCart_service_type_support_handle;
}
