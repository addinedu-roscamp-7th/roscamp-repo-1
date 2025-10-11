// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from shopee_interfaces:msg/Obstacle.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/msg/detail/obstacle__rosidl_typesupport_fastrtps_cpp.hpp"
#include "shopee_interfaces/msg/detail/obstacle__functions.h"
#include "shopee_interfaces/msg/detail/obstacle__struct.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions
namespace shopee_interfaces
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const shopee_interfaces::msg::Point2D &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  shopee_interfaces::msg::Point2D &);
size_t get_serialized_size(
  const shopee_interfaces::msg::Point2D &,
  size_t current_alignment);
size_t
max_serialized_size_Point2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
bool cdr_serialize_key(
  const shopee_interfaces::msg::Point2D &,
  eprosima::fastcdr::Cdr &);
size_t get_serialized_size_key(
  const shopee_interfaces::msg::Point2D &,
  size_t current_alignment);
size_t
max_serialized_size_key_Point2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace shopee_interfaces

namespace shopee_interfaces
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const shopee_interfaces::msg::Vector2D &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  shopee_interfaces::msg::Vector2D &);
size_t get_serialized_size(
  const shopee_interfaces::msg::Vector2D &,
  size_t current_alignment);
size_t
max_serialized_size_Vector2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
bool cdr_serialize_key(
  const shopee_interfaces::msg::Vector2D &,
  eprosima::fastcdr::Cdr &);
size_t get_serialized_size_key(
  const shopee_interfaces::msg::Vector2D &,
  size_t current_alignment);
size_t
max_serialized_size_key_Vector2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace shopee_interfaces

namespace shopee_interfaces
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const shopee_interfaces::msg::BBox &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  shopee_interfaces::msg::BBox &);
size_t get_serialized_size(
  const shopee_interfaces::msg::BBox &,
  size_t current_alignment);
size_t
max_serialized_size_BBox(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
bool cdr_serialize_key(
  const shopee_interfaces::msg::BBox &,
  eprosima::fastcdr::Cdr &);
size_t get_serialized_size_key(
  const shopee_interfaces::msg::BBox &,
  size_t current_alignment);
size_t
max_serialized_size_key_BBox(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{


bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
cdr_serialize(
  const shopee_interfaces::msg::Obstacle & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: obstacle_type
  cdr << ros_message.obstacle_type;

  // Member: position
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.position,
    cdr);

  // Member: distance
  cdr << ros_message.distance;

  // Member: velocity
  cdr << ros_message.velocity;

  // Member: direction
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.direction,
    cdr);

  // Member: bbox
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.bbox,
    cdr);

  // Member: confidence
  cdr << ros_message.confidence;

  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces::msg::Obstacle & ros_message)
{
  // Member: obstacle_type
  cdr >> ros_message.obstacle_type;

  // Member: position
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.position);

  // Member: distance
  cdr >> ros_message.distance;

  // Member: velocity
  cdr >> ros_message.velocity;

  // Member: direction
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.direction);

  // Member: bbox
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.bbox);

  // Member: confidence
  cdr >> ros_message.confidence;

  return true;
}  // NOLINT(readability/fn_size)


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
get_serialized_size(
  const shopee_interfaces::msg::Obstacle & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: obstacle_type
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.obstacle_type.size() + 1);

  // Member: position
  current_alignment +=
    shopee_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.position, current_alignment);

  // Member: distance
  {
    size_t item_size = sizeof(ros_message.distance);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Member: velocity
  {
    size_t item_size = sizeof(ros_message.velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Member: direction
  current_alignment +=
    shopee_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.direction, current_alignment);

  // Member: bbox
  current_alignment +=
    shopee_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.bbox, current_alignment);

  // Member: confidence
  {
    size_t item_size = sizeof(ros_message.confidence);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
max_serialized_size_Obstacle(
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

  // Member: obstacle_type
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
  // Member: position
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        shopee_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_Point2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // Member: distance
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // Member: velocity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // Member: direction
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        shopee_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_Vector2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // Member: bbox
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        shopee_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_BBox(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // Member: confidence
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
    using DataType = shopee_interfaces::msg::Obstacle;
    is_plain =
      (
      offsetof(DataType, confidence) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
cdr_serialize_key(
  const shopee_interfaces::msg::Obstacle & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: obstacle_type
  cdr << ros_message.obstacle_type;

  // Member: position
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize_key(
    ros_message.position,
    cdr);

  // Member: distance
  cdr << ros_message.distance;

  // Member: velocity
  cdr << ros_message.velocity;

  // Member: direction
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize_key(
    ros_message.direction,
    cdr);

  // Member: bbox
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize_key(
    ros_message.bbox,
    cdr);

  // Member: confidence
  cdr << ros_message.confidence;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
get_serialized_size_key(
  const shopee_interfaces::msg::Obstacle & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: obstacle_type
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.obstacle_type.size() + 1);

  // Member: position
  current_alignment +=
    shopee_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size_key(
    ros_message.position, current_alignment);

  // Member: distance
  {
    size_t item_size = sizeof(ros_message.distance);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Member: velocity
  {
    size_t item_size = sizeof(ros_message.velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Member: direction
  current_alignment +=
    shopee_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size_key(
    ros_message.direction, current_alignment);

  // Member: bbox
  current_alignment +=
    shopee_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size_key(
    ros_message.bbox, current_alignment);

  // Member: confidence
  {
    size_t item_size = sizeof(ros_message.confidence);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
max_serialized_size_key_Obstacle(
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

  // Member: obstacle_type
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

  // Member: position
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        shopee_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_key_Point2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: distance
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: velocity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: direction
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        shopee_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_key_Vector2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: bbox
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        shopee_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_key_BBox(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: confidence
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
    using DataType = shopee_interfaces::msg::Obstacle;
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
  auto typed_message =
    static_cast<const shopee_interfaces::msg::Obstacle *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _Obstacle__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<shopee_interfaces::msg::Obstacle *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _Obstacle__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const shopee_interfaces::msg::Obstacle *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _Obstacle__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_Obstacle(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _Obstacle__callbacks = {
  "shopee_interfaces::msg",
  "Obstacle",
  _Obstacle__cdr_serialize,
  _Obstacle__cdr_deserialize,
  _Obstacle__get_serialized_size,
  _Obstacle__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _Obstacle__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_Obstacle__callbacks,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__Obstacle__get_type_hash,
  &shopee_interfaces__msg__Obstacle__get_type_description,
  &shopee_interfaces__msg__Obstacle__get_type_description_sources,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace shopee_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_shopee_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<shopee_interfaces::msg::Obstacle>()
{
  return &shopee_interfaces::msg::typesupport_fastrtps_cpp::_Obstacle__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, shopee_interfaces, msg, Obstacle)() {
  return &shopee_interfaces::msg::typesupport_fastrtps_cpp::_Obstacle__handle;
}

#ifdef __cplusplus
}
#endif
