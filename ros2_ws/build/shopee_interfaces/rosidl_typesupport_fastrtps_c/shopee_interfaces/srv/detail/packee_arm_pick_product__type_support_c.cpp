// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from shopee_interfaces:srv/PackeeArmPickProduct.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/srv/detail/packee_arm_pick_product__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <cstddef>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "shopee_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "shopee_interfaces/srv/detail/packee_arm_pick_product__struct.h"
#include "shopee_interfaces/srv/detail/packee_arm_pick_product__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "rosidl_runtime_c/string.h"  // arm_side
#include "rosidl_runtime_c/string_functions.h"  // arm_side
#include "shopee_interfaces/msg/detail/b_box__functions.h"  // bbox
#include "shopee_interfaces/msg/detail/point3_d__functions.h"  // target_position

// forward declare type support functions

bool cdr_serialize_shopee_interfaces__msg__BBox(
  const shopee_interfaces__msg__BBox * ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool cdr_deserialize_shopee_interfaces__msg__BBox(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__msg__BBox * ros_message);

size_t get_serialized_size_shopee_interfaces__msg__BBox(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_shopee_interfaces__msg__BBox(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool cdr_serialize_key_shopee_interfaces__msg__BBox(
  const shopee_interfaces__msg__BBox * ros_message,
  eprosima::fastcdr::Cdr & cdr);

size_t get_serialized_size_key_shopee_interfaces__msg__BBox(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_key_shopee_interfaces__msg__BBox(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, msg, BBox)();

bool cdr_serialize_shopee_interfaces__msg__Point3D(
  const shopee_interfaces__msg__Point3D * ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool cdr_deserialize_shopee_interfaces__msg__Point3D(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__msg__Point3D * ros_message);

size_t get_serialized_size_shopee_interfaces__msg__Point3D(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_shopee_interfaces__msg__Point3D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool cdr_serialize_key_shopee_interfaces__msg__Point3D(
  const shopee_interfaces__msg__Point3D * ros_message,
  eprosima::fastcdr::Cdr & cdr);

size_t get_serialized_size_key_shopee_interfaces__msg__Point3D(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_key_shopee_interfaces__msg__Point3D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, msg, Point3D)();


using _PackeeArmPickProduct_Request__ros_msg_type = shopee_interfaces__srv__PackeeArmPickProduct_Request;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  const shopee_interfaces__srv__PackeeArmPickProduct_Request * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: robot_id
  {
    cdr << ros_message->robot_id;
  }

  // Field name: order_id
  {
    cdr << ros_message->order_id;
  }

  // Field name: product_id
  {
    cdr << ros_message->product_id;
  }

  // Field name: arm_side
  {
    const rosidl_runtime_c__String * str = &ros_message->arm_side;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: target_position
  {
    cdr_serialize_shopee_interfaces__msg__Point3D(
      &ros_message->target_position, cdr);
  }

  // Field name: bbox
  {
    cdr_serialize_shopee_interfaces__msg__BBox(
      &ros_message->bbox, cdr);
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__srv__PackeeArmPickProduct_Request * ros_message)
{
  // Field name: robot_id
  {
    cdr >> ros_message->robot_id;
  }

  // Field name: order_id
  {
    cdr >> ros_message->order_id;
  }

  // Field name: product_id
  {
    cdr >> ros_message->product_id;
  }

  // Field name: arm_side
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->arm_side.data) {
      rosidl_runtime_c__String__init(&ros_message->arm_side);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->arm_side,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'arm_side'\n");
      return false;
    }
  }

  // Field name: target_position
  {
    cdr_deserialize_shopee_interfaces__msg__Point3D(cdr, &ros_message->target_position);
  }

  // Field name: bbox
  {
    cdr_deserialize_shopee_interfaces__msg__BBox(cdr, &ros_message->bbox);
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PackeeArmPickProduct_Request__ros_msg_type * ros_message = static_cast<const _PackeeArmPickProduct_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: robot_id
  {
    size_t item_size = sizeof(ros_message->robot_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: order_id
  {
    size_t item_size = sizeof(ros_message->order_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: product_id
  {
    size_t item_size = sizeof(ros_message->product_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: arm_side
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->arm_side.size + 1);

  // Field name: target_position
  current_alignment += get_serialized_size_shopee_interfaces__msg__Point3D(
    &(ros_message->target_position), current_alignment);

  // Field name: bbox
  current_alignment += get_serialized_size_shopee_interfaces__msg__BBox(
    &(ros_message->bbox), current_alignment);

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Field name: robot_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: order_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: product_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: arm_side
  {
    size_t array_size = 1;
    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  // Field name: target_position
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_shopee_interfaces__msg__Point3D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: bbox
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_shopee_interfaces__msg__BBox(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = shopee_interfaces__srv__PackeeArmPickProduct_Request;
    is_plain =
      (
      offsetof(DataType, bbox) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  const shopee_interfaces__srv__PackeeArmPickProduct_Request * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: robot_id
  {
    cdr << ros_message->robot_id;
  }

  // Field name: order_id
  {
    cdr << ros_message->order_id;
  }

  // Field name: product_id
  {
    cdr << ros_message->product_id;
  }

  // Field name: arm_side
  {
    const rosidl_runtime_c__String * str = &ros_message->arm_side;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: target_position
  {
    cdr_serialize_key_shopee_interfaces__msg__Point3D(
      &ros_message->target_position, cdr);
  }

  // Field name: bbox
  {
    cdr_serialize_key_shopee_interfaces__msg__BBox(
      &ros_message->bbox, cdr);
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PackeeArmPickProduct_Request__ros_msg_type * ros_message = static_cast<const _PackeeArmPickProduct_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: robot_id
  {
    size_t item_size = sizeof(ros_message->robot_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: order_id
  {
    size_t item_size = sizeof(ros_message->order_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: product_id
  {
    size_t item_size = sizeof(ros_message->product_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: arm_side
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->arm_side.size + 1);

  // Field name: target_position
  current_alignment += get_serialized_size_key_shopee_interfaces__msg__Point3D(
    &(ros_message->target_position), current_alignment);

  // Field name: bbox
  current_alignment += get_serialized_size_key_shopee_interfaces__msg__BBox(
    &(ros_message->bbox), current_alignment);

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;
  // Field name: robot_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: order_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: product_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: arm_side
  {
    size_t array_size = 1;
    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  // Field name: target_position
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_shopee_interfaces__msg__Point3D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: bbox
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_shopee_interfaces__msg__BBox(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = shopee_interfaces__srv__PackeeArmPickProduct_Request;
    is_plain =
      (
      offsetof(DataType, bbox) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _PackeeArmPickProduct_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const shopee_interfaces__srv__PackeeArmPickProduct_Request * ros_message = static_cast<const shopee_interfaces__srv__PackeeArmPickProduct_Request *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Request(ros_message, cdr);
}

static bool _PackeeArmPickProduct_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  shopee_interfaces__srv__PackeeArmPickProduct_Request * ros_message = static_cast<shopee_interfaces__srv__PackeeArmPickProduct_Request *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Request(cdr, ros_message);
}

static uint32_t _PackeeArmPickProduct_Request__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Request(
      untyped_ros_message, 0));
}

static size_t _PackeeArmPickProduct_Request__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Request(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_PackeeArmPickProduct_Request = {
  "shopee_interfaces::srv",
  "PackeeArmPickProduct_Request",
  _PackeeArmPickProduct_Request__cdr_serialize,
  _PackeeArmPickProduct_Request__cdr_deserialize,
  _PackeeArmPickProduct_Request__get_serialized_size,
  _PackeeArmPickProduct_Request__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _PackeeArmPickProduct_Request__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PackeeArmPickProduct_Request,
  get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PackeeArmPickProduct_Request__get_type_hash,
  &shopee_interfaces__srv__PackeeArmPickProduct_Request__get_type_description,
  &shopee_interfaces__srv__PackeeArmPickProduct_Request__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, srv, PackeeArmPickProduct_Request)() {
  return &_PackeeArmPickProduct_Request__type_support;
}

#if defined(__cplusplus)
}
#endif

// already included above
// #include <cassert>
// already included above
// #include <cstddef>
// already included above
// #include <limits>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "shopee_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_arm_pick_product__struct.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_arm_pick_product__functions.h"
// already included above
// #include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

// already included above
// #include "rosidl_runtime_c/string.h"  // message
// already included above
// #include "rosidl_runtime_c/string_functions.h"  // message

// forward declare type support functions


using _PackeeArmPickProduct_Response__ros_msg_type = shopee_interfaces__srv__PackeeArmPickProduct_Response;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  const shopee_interfaces__srv__PackeeArmPickProduct_Response * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: accepted
  {
    cdr << (ros_message->accepted ? true : false);
  }

  // Field name: message
  {
    const rosidl_runtime_c__String * str = &ros_message->message;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__srv__PackeeArmPickProduct_Response * ros_message)
{
  // Field name: accepted
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->accepted = tmp ? true : false;
  }

  // Field name: message
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->message.data) {
      rosidl_runtime_c__String__init(&ros_message->message);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->message,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'message'\n");
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PackeeArmPickProduct_Response__ros_msg_type * ros_message = static_cast<const _PackeeArmPickProduct_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: accepted
  {
    size_t item_size = sizeof(ros_message->accepted);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: message
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->message.size + 1);

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Field name: accepted
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Field name: message
  {
    size_t array_size = 1;
    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = shopee_interfaces__srv__PackeeArmPickProduct_Response;
    is_plain =
      (
      offsetof(DataType, message) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  const shopee_interfaces__srv__PackeeArmPickProduct_Response * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: accepted
  {
    cdr << (ros_message->accepted ? true : false);
  }

  // Field name: message
  {
    const rosidl_runtime_c__String * str = &ros_message->message;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PackeeArmPickProduct_Response__ros_msg_type * ros_message = static_cast<const _PackeeArmPickProduct_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: accepted
  {
    size_t item_size = sizeof(ros_message->accepted);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: message
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->message.size + 1);

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;
  // Field name: accepted
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Field name: message
  {
    size_t array_size = 1;
    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = shopee_interfaces__srv__PackeeArmPickProduct_Response;
    is_plain =
      (
      offsetof(DataType, message) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _PackeeArmPickProduct_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const shopee_interfaces__srv__PackeeArmPickProduct_Response * ros_message = static_cast<const shopee_interfaces__srv__PackeeArmPickProduct_Response *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Response(ros_message, cdr);
}

static bool _PackeeArmPickProduct_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  shopee_interfaces__srv__PackeeArmPickProduct_Response * ros_message = static_cast<shopee_interfaces__srv__PackeeArmPickProduct_Response *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Response(cdr, ros_message);
}

static uint32_t _PackeeArmPickProduct_Response__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Response(
      untyped_ros_message, 0));
}

static size_t _PackeeArmPickProduct_Response__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Response(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_PackeeArmPickProduct_Response = {
  "shopee_interfaces::srv",
  "PackeeArmPickProduct_Response",
  _PackeeArmPickProduct_Response__cdr_serialize,
  _PackeeArmPickProduct_Response__cdr_deserialize,
  _PackeeArmPickProduct_Response__get_serialized_size,
  _PackeeArmPickProduct_Response__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _PackeeArmPickProduct_Response__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PackeeArmPickProduct_Response,
  get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PackeeArmPickProduct_Response__get_type_hash,
  &shopee_interfaces__srv__PackeeArmPickProduct_Response__get_type_description,
  &shopee_interfaces__srv__PackeeArmPickProduct_Response__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, srv, PackeeArmPickProduct_Response)() {
  return &_PackeeArmPickProduct_Response__type_support;
}

#if defined(__cplusplus)
}
#endif

// already included above
// #include <cassert>
// already included above
// #include <cstddef>
// already included above
// #include <limits>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "shopee_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_arm_pick_product__struct.h"
// already included above
// #include "shopee_interfaces/srv/detail/packee_arm_pick_product__functions.h"
// already included above
// #include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "service_msgs/msg/detail/service_event_info__functions.h"  // info

// forward declare type support functions

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_shopee_interfaces
bool cdr_serialize_service_msgs__msg__ServiceEventInfo(
  const service_msgs__msg__ServiceEventInfo * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_shopee_interfaces
bool cdr_deserialize_service_msgs__msg__ServiceEventInfo(
  eprosima::fastcdr::Cdr & cdr,
  service_msgs__msg__ServiceEventInfo * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_shopee_interfaces
size_t get_serialized_size_service_msgs__msg__ServiceEventInfo(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_shopee_interfaces
size_t max_serialized_size_service_msgs__msg__ServiceEventInfo(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_shopee_interfaces
bool cdr_serialize_key_service_msgs__msg__ServiceEventInfo(
  const service_msgs__msg__ServiceEventInfo * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_shopee_interfaces
size_t get_serialized_size_key_service_msgs__msg__ServiceEventInfo(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_shopee_interfaces
size_t max_serialized_size_key_service_msgs__msg__ServiceEventInfo(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_shopee_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, service_msgs, msg, ServiceEventInfo)();

bool cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  const shopee_interfaces__srv__PackeeArmPickProduct_Request * ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__srv__PackeeArmPickProduct_Request * ros_message);

size_t get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool cdr_serialize_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  const shopee_interfaces__srv__PackeeArmPickProduct_Request * ros_message,
  eprosima::fastcdr::Cdr & cdr);

size_t get_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, srv, PackeeArmPickProduct_Request)();

bool cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  const shopee_interfaces__srv__PackeeArmPickProduct_Response * ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__srv__PackeeArmPickProduct_Response * ros_message);

size_t get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool cdr_serialize_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  const shopee_interfaces__srv__PackeeArmPickProduct_Response * ros_message,
  eprosima::fastcdr::Cdr & cdr);

size_t get_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, srv, PackeeArmPickProduct_Response)();


using _PackeeArmPickProduct_Event__ros_msg_type = shopee_interfaces__srv__PackeeArmPickProduct_Event;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Event(
  const shopee_interfaces__srv__PackeeArmPickProduct_Event * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: info
  {
    cdr_serialize_service_msgs__msg__ServiceEventInfo(
      &ros_message->info, cdr);
  }

  // Field name: request
  {
    size_t size = ros_message->request.size;
    auto array_ptr = ros_message->request.data;
    if (size > 1) {
      fprintf(stderr, "array size exceeds upper bound\n");
      return false;
    }
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Request(
        &array_ptr[i], cdr);
    }
  }

  // Field name: response
  {
    size_t size = ros_message->response.size;
    auto array_ptr = ros_message->response.data;
    if (size > 1) {
      fprintf(stderr, "array size exceeds upper bound\n");
      return false;
    }
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Response(
        &array_ptr[i], cdr);
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Event(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__srv__PackeeArmPickProduct_Event * ros_message)
{
  // Field name: info
  {
    cdr_deserialize_service_msgs__msg__ServiceEventInfo(cdr, &ros_message->info);
  }

  // Field name: request
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.get_state();
    bool correct_size = cdr.jump(size);
    cdr.set_state(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->request.data) {
      shopee_interfaces__srv__PackeeArmPickProduct_Request__Sequence__fini(&ros_message->request);
    }
    if (!shopee_interfaces__srv__PackeeArmPickProduct_Request__Sequence__init(&ros_message->request, size)) {
      fprintf(stderr, "failed to create array for field 'request'");
      return false;
    }
    auto array_ptr = ros_message->request.data;
    for (size_t i = 0; i < size; ++i) {
      cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Request(cdr, &array_ptr[i]);
    }
  }

  // Field name: response
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.get_state();
    bool correct_size = cdr.jump(size);
    cdr.set_state(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->response.data) {
      shopee_interfaces__srv__PackeeArmPickProduct_Response__Sequence__fini(&ros_message->response);
    }
    if (!shopee_interfaces__srv__PackeeArmPickProduct_Response__Sequence__init(&ros_message->response, size)) {
      fprintf(stderr, "failed to create array for field 'response'");
      return false;
    }
    auto array_ptr = ros_message->response.data;
    for (size_t i = 0; i < size; ++i) {
      cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Response(cdr, &array_ptr[i]);
    }
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Event(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PackeeArmPickProduct_Event__ros_msg_type * ros_message = static_cast<const _PackeeArmPickProduct_Event__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: info
  current_alignment += get_serialized_size_service_msgs__msg__ServiceEventInfo(
    &(ros_message->info), current_alignment);

  // Field name: request
  {
    size_t array_size = ros_message->request.size;
    auto array_ptr = ros_message->request.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Request(
        &array_ptr[index], current_alignment);
    }
  }

  // Field name: response
  {
    size_t array_size = ros_message->response.size;
    auto array_ptr = ros_message->response.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Response(
        &array_ptr[index], current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Event(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Field name: info
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_service_msgs__msg__ServiceEventInfo(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: request
  {
    size_t array_size = 1;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Request(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: response
  {
    size_t array_size = 1;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Response(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = shopee_interfaces__srv__PackeeArmPickProduct_Event;
    is_plain =
      (
      offsetof(DataType, response) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_key_shopee_interfaces__srv__PackeeArmPickProduct_Event(
  const shopee_interfaces__srv__PackeeArmPickProduct_Event * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: info
  {
    cdr_serialize_key_service_msgs__msg__ServiceEventInfo(
      &ros_message->info, cdr);
  }

  // Field name: request
  {
    size_t size = ros_message->request.size;
    auto array_ptr = ros_message->request.data;
    if (size > 1) {
      fprintf(stderr, "array size exceeds upper bound\n");
      return false;
    }
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      cdr_serialize_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
        &array_ptr[i], cdr);
    }
  }

  // Field name: response
  {
    size_t size = ros_message->response.size;
    auto array_ptr = ros_message->response.data;
    if (size > 1) {
      fprintf(stderr, "array size exceeds upper bound\n");
      return false;
    }
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      cdr_serialize_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
        &array_ptr[i], cdr);
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Event(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PackeeArmPickProduct_Event__ros_msg_type * ros_message = static_cast<const _PackeeArmPickProduct_Event__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: info
  current_alignment += get_serialized_size_key_service_msgs__msg__ServiceEventInfo(
    &(ros_message->info), current_alignment);

  // Field name: request
  {
    size_t array_size = ros_message->request.size;
    auto array_ptr = ros_message->request.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
        &array_ptr[index], current_alignment);
    }
  }

  // Field name: response
  {
    size_t array_size = ros_message->response.size;
    auto array_ptr = ros_message->response.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
        &array_ptr[index], current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Event(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;
  // Field name: info
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_service_msgs__msg__ServiceEventInfo(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: request
  {
    size_t array_size = 1;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Request(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: response
  {
    size_t array_size = 1;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_shopee_interfaces__srv__PackeeArmPickProduct_Response(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = shopee_interfaces__srv__PackeeArmPickProduct_Event;
    is_plain =
      (
      offsetof(DataType, response) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _PackeeArmPickProduct_Event__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const shopee_interfaces__srv__PackeeArmPickProduct_Event * ros_message = static_cast<const shopee_interfaces__srv__PackeeArmPickProduct_Event *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_shopee_interfaces__srv__PackeeArmPickProduct_Event(ros_message, cdr);
}

static bool _PackeeArmPickProduct_Event__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  shopee_interfaces__srv__PackeeArmPickProduct_Event * ros_message = static_cast<shopee_interfaces__srv__PackeeArmPickProduct_Event *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_shopee_interfaces__srv__PackeeArmPickProduct_Event(cdr, ros_message);
}

static uint32_t _PackeeArmPickProduct_Event__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Event(
      untyped_ros_message, 0));
}

static size_t _PackeeArmPickProduct_Event__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_shopee_interfaces__srv__PackeeArmPickProduct_Event(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_PackeeArmPickProduct_Event = {
  "shopee_interfaces::srv",
  "PackeeArmPickProduct_Event",
  _PackeeArmPickProduct_Event__cdr_serialize,
  _PackeeArmPickProduct_Event__cdr_deserialize,
  _PackeeArmPickProduct_Event__get_serialized_size,
  _PackeeArmPickProduct_Event__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _PackeeArmPickProduct_Event__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PackeeArmPickProduct_Event,
  get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PackeeArmPickProduct_Event__get_type_hash,
  &shopee_interfaces__srv__PackeeArmPickProduct_Event__get_type_description,
  &shopee_interfaces__srv__PackeeArmPickProduct_Event__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, srv, PackeeArmPickProduct_Event)() {
  return &_PackeeArmPickProduct_Event__type_support;
}

#if defined(__cplusplus)
}
#endif

#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "shopee_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "shopee_interfaces/srv/packee_arm_pick_product.h"

#if defined(__cplusplus)
extern "C"
{
#endif

static service_type_support_callbacks_t PackeeArmPickProduct__callbacks = {
  "shopee_interfaces::srv",
  "PackeeArmPickProduct",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, srv, PackeeArmPickProduct_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, srv, PackeeArmPickProduct_Response)(),
};

static rosidl_service_type_support_t PackeeArmPickProduct__handle = {
  rosidl_typesupport_fastrtps_c__identifier,
  &PackeeArmPickProduct__callbacks,
  get_service_typesupport_handle_function,
  &_PackeeArmPickProduct_Request__type_support,
  &_PackeeArmPickProduct_Response__type_support,
  &_PackeeArmPickProduct_Event__type_support,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    shopee_interfaces,
    srv,
    PackeeArmPickProduct
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    shopee_interfaces,
    srv,
    PackeeArmPickProduct
  ),
  &shopee_interfaces__srv__PackeeArmPickProduct__get_type_hash,
  &shopee_interfaces__srv__PackeeArmPickProduct__get_type_description,
  &shopee_interfaces__srv__PackeeArmPickProduct__get_type_description_sources,
};

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, srv, PackeeArmPickProduct)() {
  return &PackeeArmPickProduct__handle;
}

#if defined(__cplusplus)
}
#endif
