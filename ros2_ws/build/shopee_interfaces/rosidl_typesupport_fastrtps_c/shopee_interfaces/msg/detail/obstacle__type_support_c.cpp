// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from shopee_interfaces:msg/Obstacle.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/msg/detail/obstacle__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <cstddef>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "shopee_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "shopee_interfaces/msg/detail/obstacle__struct.h"
#include "shopee_interfaces/msg/detail/obstacle__functions.h"
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

#include "rosidl_runtime_c/string.h"  // obstacle_type
#include "rosidl_runtime_c/string_functions.h"  // obstacle_type
#include "shopee_interfaces/msg/detail/b_box__functions.h"  // bbox
#include "shopee_interfaces/msg/detail/point2_d__functions.h"  // position
#include "shopee_interfaces/msg/detail/vector2_d__functions.h"  // direction

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

bool cdr_serialize_shopee_interfaces__msg__Point2D(
  const shopee_interfaces__msg__Point2D * ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool cdr_deserialize_shopee_interfaces__msg__Point2D(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__msg__Point2D * ros_message);

size_t get_serialized_size_shopee_interfaces__msg__Point2D(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_shopee_interfaces__msg__Point2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool cdr_serialize_key_shopee_interfaces__msg__Point2D(
  const shopee_interfaces__msg__Point2D * ros_message,
  eprosima::fastcdr::Cdr & cdr);

size_t get_serialized_size_key_shopee_interfaces__msg__Point2D(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_key_shopee_interfaces__msg__Point2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, msg, Point2D)();

bool cdr_serialize_shopee_interfaces__msg__Vector2D(
  const shopee_interfaces__msg__Vector2D * ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool cdr_deserialize_shopee_interfaces__msg__Vector2D(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__msg__Vector2D * ros_message);

size_t get_serialized_size_shopee_interfaces__msg__Vector2D(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_shopee_interfaces__msg__Vector2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool cdr_serialize_key_shopee_interfaces__msg__Vector2D(
  const shopee_interfaces__msg__Vector2D * ros_message,
  eprosima::fastcdr::Cdr & cdr);

size_t get_serialized_size_key_shopee_interfaces__msg__Vector2D(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_key_shopee_interfaces__msg__Vector2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, msg, Vector2D)();


using _Obstacle__ros_msg_type = shopee_interfaces__msg__Obstacle;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_shopee_interfaces__msg__Obstacle(
  const shopee_interfaces__msg__Obstacle * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: obstacle_type
  {
    const rosidl_runtime_c__String * str = &ros_message->obstacle_type;
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

  // Field name: position
  {
    cdr_serialize_shopee_interfaces__msg__Point2D(
      &ros_message->position, cdr);
  }

  // Field name: distance
  {
    cdr << ros_message->distance;
  }

  // Field name: velocity
  {
    cdr << ros_message->velocity;
  }

  // Field name: direction
  {
    cdr_serialize_shopee_interfaces__msg__Vector2D(
      &ros_message->direction, cdr);
  }

  // Field name: bbox
  {
    cdr_serialize_shopee_interfaces__msg__BBox(
      &ros_message->bbox, cdr);
  }

  // Field name: confidence
  {
    cdr << ros_message->confidence;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_deserialize_shopee_interfaces__msg__Obstacle(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces__msg__Obstacle * ros_message)
{
  // Field name: obstacle_type
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->obstacle_type.data) {
      rosidl_runtime_c__String__init(&ros_message->obstacle_type);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->obstacle_type,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'obstacle_type'\n");
      return false;
    }
  }

  // Field name: position
  {
    cdr_deserialize_shopee_interfaces__msg__Point2D(cdr, &ros_message->position);
  }

  // Field name: distance
  {
    cdr >> ros_message->distance;
  }

  // Field name: velocity
  {
    cdr >> ros_message->velocity;
  }

  // Field name: direction
  {
    cdr_deserialize_shopee_interfaces__msg__Vector2D(cdr, &ros_message->direction);
  }

  // Field name: bbox
  {
    cdr_deserialize_shopee_interfaces__msg__BBox(cdr, &ros_message->bbox);
  }

  // Field name: confidence
  {
    cdr >> ros_message->confidence;
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_shopee_interfaces__msg__Obstacle(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Obstacle__ros_msg_type * ros_message = static_cast<const _Obstacle__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: obstacle_type
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->obstacle_type.size + 1);

  // Field name: position
  current_alignment += get_serialized_size_shopee_interfaces__msg__Point2D(
    &(ros_message->position), current_alignment);

  // Field name: distance
  {
    size_t item_size = sizeof(ros_message->distance);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: velocity
  {
    size_t item_size = sizeof(ros_message->velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: direction
  current_alignment += get_serialized_size_shopee_interfaces__msg__Vector2D(
    &(ros_message->direction), current_alignment);

  // Field name: bbox
  current_alignment += get_serialized_size_shopee_interfaces__msg__BBox(
    &(ros_message->bbox), current_alignment);

  // Field name: confidence
  {
    size_t item_size = sizeof(ros_message->confidence);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_shopee_interfaces__msg__Obstacle(
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

  // Field name: obstacle_type
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

  // Field name: position
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_shopee_interfaces__msg__Point2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: distance
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: velocity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: direction
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_shopee_interfaces__msg__Vector2D(
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

  // Field name: confidence
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = shopee_interfaces__msg__Obstacle;
    is_plain =
      (
      offsetof(DataType, confidence) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_key_shopee_interfaces__msg__Obstacle(
  const shopee_interfaces__msg__Obstacle * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: obstacle_type
  {
    const rosidl_runtime_c__String * str = &ros_message->obstacle_type;
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

  // Field name: position
  {
    cdr_serialize_key_shopee_interfaces__msg__Point2D(
      &ros_message->position, cdr);
  }

  // Field name: distance
  {
    cdr << ros_message->distance;
  }

  // Field name: velocity
  {
    cdr << ros_message->velocity;
  }

  // Field name: direction
  {
    cdr_serialize_key_shopee_interfaces__msg__Vector2D(
      &ros_message->direction, cdr);
  }

  // Field name: bbox
  {
    cdr_serialize_key_shopee_interfaces__msg__BBox(
      &ros_message->bbox, cdr);
  }

  // Field name: confidence
  {
    cdr << ros_message->confidence;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_key_shopee_interfaces__msg__Obstacle(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Obstacle__ros_msg_type * ros_message = static_cast<const _Obstacle__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: obstacle_type
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->obstacle_type.size + 1);

  // Field name: position
  current_alignment += get_serialized_size_key_shopee_interfaces__msg__Point2D(
    &(ros_message->position), current_alignment);

  // Field name: distance
  {
    size_t item_size = sizeof(ros_message->distance);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: velocity
  {
    size_t item_size = sizeof(ros_message->velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: direction
  current_alignment += get_serialized_size_key_shopee_interfaces__msg__Vector2D(
    &(ros_message->direction), current_alignment);

  // Field name: bbox
  current_alignment += get_serialized_size_key_shopee_interfaces__msg__BBox(
    &(ros_message->bbox), current_alignment);

  // Field name: confidence
  {
    size_t item_size = sizeof(ros_message->confidence);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_key_shopee_interfaces__msg__Obstacle(
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
  // Field name: obstacle_type
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

  // Field name: position
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_shopee_interfaces__msg__Point2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: distance
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: velocity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: direction
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_shopee_interfaces__msg__Vector2D(
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

  // Field name: confidence
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = shopee_interfaces__msg__Obstacle;
    is_plain =
      (
      offsetof(DataType, confidence) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _Obstacle__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const shopee_interfaces__msg__Obstacle * ros_message = static_cast<const shopee_interfaces__msg__Obstacle *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_shopee_interfaces__msg__Obstacle(ros_message, cdr);
}

static bool _Obstacle__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  shopee_interfaces__msg__Obstacle * ros_message = static_cast<shopee_interfaces__msg__Obstacle *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_shopee_interfaces__msg__Obstacle(cdr, ros_message);
}

static uint32_t _Obstacle__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_shopee_interfaces__msg__Obstacle(
      untyped_ros_message, 0));
}

static size_t _Obstacle__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_shopee_interfaces__msg__Obstacle(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_Obstacle = {
  "shopee_interfaces::msg",
  "Obstacle",
  _Obstacle__cdr_serialize,
  _Obstacle__cdr_deserialize,
  _Obstacle__get_serialized_size,
  _Obstacle__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _Obstacle__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Obstacle,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__Obstacle__get_type_hash,
  &shopee_interfaces__msg__Obstacle__get_type_description,
  &shopee_interfaces__msg__Obstacle__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, msg, Obstacle)() {
  return &_Obstacle__type_support;
}

#if defined(__cplusplus)
}
#endif
