// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from shopee_interfaces:msg/PickeeMobileArrival.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/msg/detail/pickee_mobile_arrival__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <cstddef>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "shopee_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "shopee_interfaces/msg/detail/pickee_mobile_arrival__struct.h"
#include "shopee_interfaces/msg/detail/pickee_mobile_arrival__functions.h"
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

#include "rosidl_runtime_c/string.h"  // message
#include "rosidl_runtime_c/string_functions.h"  // message
#include "shopee_interfaces/msg/detail/pose2_d__functions.h"  // final_pose

// forward declare type support functions

bool cdr_serialize_shopee_interfaces__msg__Pose2D(
  const shopee_interfaces__msg__Pose2D * ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool cdr_deserialize_shopee_interfaces__msg__Pose2D(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__msg__Pose2D * ros_message);

size_t get_serialized_size_shopee_interfaces__msg__Pose2D(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_shopee_interfaces__msg__Pose2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool cdr_serialize_key_shopee_interfaces__msg__Pose2D(
  const shopee_interfaces__msg__Pose2D * ros_message,
  eprosima::fastcdr::Cdr & cdr);

size_t get_serialized_size_key_shopee_interfaces__msg__Pose2D(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_key_shopee_interfaces__msg__Pose2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, msg, Pose2D)();


using _PickeeMobileArrival__ros_msg_type = shopee_interfaces__msg__PickeeMobileArrival;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_shopee_interfaces__msg__PickeeMobileArrival(
  const shopee_interfaces__msg__PickeeMobileArrival * ros_message,
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

  // Field name: location_id
  {
    cdr << ros_message->location_id;
  }

  // Field name: final_pose
  {
    cdr_serialize_shopee_interfaces__msg__Pose2D(
      &ros_message->final_pose, cdr);
  }

  // Field name: position_error
  {
    cdr << ros_message->position_error;
  }

  // Field name: travel_time
  {
    cdr << ros_message->travel_time;
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
bool cdr_deserialize_shopee_interfaces__msg__PickeeMobileArrival(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__msg__PickeeMobileArrival * ros_message)
{
  // Field name: robot_id
  {
    cdr >> ros_message->robot_id;
  }

  // Field name: order_id
  {
    cdr >> ros_message->order_id;
  }

  // Field name: location_id
  {
    cdr >> ros_message->location_id;
  }

  // Field name: final_pose
  {
    cdr_deserialize_shopee_interfaces__msg__Pose2D(cdr, &ros_message->final_pose);
  }

  // Field name: position_error
  {
    cdr >> ros_message->position_error;
  }

  // Field name: travel_time
  {
    cdr >> ros_message->travel_time;
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
size_t get_serialized_size_shopee_interfaces__msg__PickeeMobileArrival(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PickeeMobileArrival__ros_msg_type * ros_message = static_cast<const _PickeeMobileArrival__ros_msg_type *>(untyped_ros_message);
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

  // Field name: location_id
  {
    size_t item_size = sizeof(ros_message->location_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: final_pose
  current_alignment += get_serialized_size_shopee_interfaces__msg__Pose2D(
    &(ros_message->final_pose), current_alignment);

  // Field name: position_error
  {
    size_t item_size = sizeof(ros_message->position_error);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: travel_time
  {
    size_t item_size = sizeof(ros_message->travel_time);
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
size_t max_serialized_size_shopee_interfaces__msg__PickeeMobileArrival(
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

  // Field name: location_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: final_pose
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_shopee_interfaces__msg__Pose2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: position_error
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: travel_time
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
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
    using DataType = shopee_interfaces__msg__PickeeMobileArrival;
    is_plain =
      (
      offsetof(DataType, message) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_key_shopee_interfaces__msg__PickeeMobileArrival(
  const shopee_interfaces__msg__PickeeMobileArrival * ros_message,
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

  // Field name: location_id
  {
    cdr << ros_message->location_id;
  }

  // Field name: final_pose
  {
    cdr_serialize_key_shopee_interfaces__msg__Pose2D(
      &ros_message->final_pose, cdr);
  }

  // Field name: position_error
  {
    cdr << ros_message->position_error;
  }

  // Field name: travel_time
  {
    cdr << ros_message->travel_time;
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
size_t get_serialized_size_key_shopee_interfaces__msg__PickeeMobileArrival(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PickeeMobileArrival__ros_msg_type * ros_message = static_cast<const _PickeeMobileArrival__ros_msg_type *>(untyped_ros_message);
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

  // Field name: location_id
  {
    size_t item_size = sizeof(ros_message->location_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: final_pose
  current_alignment += get_serialized_size_key_shopee_interfaces__msg__Pose2D(
    &(ros_message->final_pose), current_alignment);

  // Field name: position_error
  {
    size_t item_size = sizeof(ros_message->position_error);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: travel_time
  {
    size_t item_size = sizeof(ros_message->travel_time);
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
size_t max_serialized_size_key_shopee_interfaces__msg__PickeeMobileArrival(
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

  // Field name: location_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: final_pose
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_shopee_interfaces__msg__Pose2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: position_error
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: travel_time
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
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
    using DataType = shopee_interfaces__msg__PickeeMobileArrival;
    is_plain =
      (
      offsetof(DataType, message) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _PickeeMobileArrival__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const shopee_interfaces__msg__PickeeMobileArrival * ros_message = static_cast<const shopee_interfaces__msg__PickeeMobileArrival *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_shopee_interfaces__msg__PickeeMobileArrival(ros_message, cdr);
}

static bool _PickeeMobileArrival__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  shopee_interfaces__msg__PickeeMobileArrival * ros_message = static_cast<shopee_interfaces__msg__PickeeMobileArrival *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_shopee_interfaces__msg__PickeeMobileArrival(cdr, ros_message);
}

static uint32_t _PickeeMobileArrival__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_shopee_interfaces__msg__PickeeMobileArrival(
      untyped_ros_message, 0));
}

static size_t _PickeeMobileArrival__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_shopee_interfaces__msg__PickeeMobileArrival(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_PickeeMobileArrival = {
  "shopee_interfaces::msg",
  "PickeeMobileArrival",
  _PickeeMobileArrival__cdr_serialize,
  _PickeeMobileArrival__cdr_deserialize,
  _PickeeMobileArrival__get_serialized_size,
  _PickeeMobileArrival__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _PickeeMobileArrival__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PickeeMobileArrival,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__PickeeMobileArrival__get_type_hash,
  &shopee_interfaces__msg__PickeeMobileArrival__get_type_description,
  &shopee_interfaces__msg__PickeeMobileArrival__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, msg, PickeeMobileArrival)() {
  return &_PickeeMobileArrival__type_support;
}

#if defined(__cplusplus)
}
#endif
