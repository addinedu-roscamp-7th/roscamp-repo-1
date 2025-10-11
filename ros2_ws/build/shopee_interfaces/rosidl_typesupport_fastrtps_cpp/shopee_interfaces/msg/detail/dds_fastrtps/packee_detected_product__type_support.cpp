// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from shopee_interfaces:msg/PackeeDetectedProduct.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/msg/detail/packee_detected_product__rosidl_typesupport_fastrtps_cpp.hpp"
#include "shopee_interfaces/msg/detail/packee_detected_product__functions.h"
#include "shopee_interfaces/msg/detail/packee_detected_product__struct.hpp"

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
bool cdr_serialize(
  const shopee_interfaces::msg::Point3D &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  shopee_interfaces::msg::Point3D &);
size_t get_serialized_size(
  const shopee_interfaces::msg::Point3D &,
  size_t current_alignment);
size_t
max_serialized_size_Point3D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
bool cdr_serialize_key(
  const shopee_interfaces::msg::Point3D &,
  eprosima::fastcdr::Cdr &);
size_t get_serialized_size_key(
  const shopee_interfaces::msg::Point3D &,
  size_t current_alignment);
size_t
max_serialized_size_key_Point3D(
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
  const shopee_interfaces::msg::PackeeDetectedProduct & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: product_id
  cdr << ros_message.product_id;

  // Member: bbox
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.bbox,
    cdr);

  // Member: confidence
  cdr << ros_message.confidence;

  // Member: position
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.position,
    cdr);

  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  shopee_interfaces::msg::PackeeDetectedProduct & ros_message)
{
  // Member: product_id
  cdr >> ros_message.product_id;

  // Member: bbox
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.bbox);

  // Member: confidence
  cdr >> ros_message.confidence;

  // Member: position
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.position);

  return true;
}  // NOLINT(readability/fn_size)


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
get_serialized_size(
  const shopee_interfaces::msg::PackeeDetectedProduct & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: product_id
  {
    size_t item_size = sizeof(ros_message.product_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

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

  // Member: position
  current_alignment +=
    shopee_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.position, current_alignment);

  return current_alignment - initial_alignment;
}


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
max_serialized_size_PackeeDetectedProduct(
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

  // Member: product_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
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
  // Member: position
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        shopee_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_Point3D(
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
    using DataType = shopee_interfaces::msg::PackeeDetectedProduct;
    is_plain =
      (
      offsetof(DataType, position) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
cdr_serialize_key(
  const shopee_interfaces::msg::PackeeDetectedProduct & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: product_id
  cdr << ros_message.product_id;

  // Member: bbox
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize_key(
    ros_message.bbox,
    cdr);

  // Member: confidence
  cdr << ros_message.confidence;

  // Member: position
  shopee_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize_key(
    ros_message.position,
    cdr);

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
get_serialized_size_key(
  const shopee_interfaces::msg::PackeeDetectedProduct & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: product_id
  {
    size_t item_size = sizeof(ros_message.product_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

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

  // Member: position
  current_alignment +=
    shopee_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size_key(
    ros_message.position, current_alignment);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_shopee_interfaces
max_serialized_size_key_PackeeDetectedProduct(
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

  // Member: product_id
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
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

  // Member: position
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        shopee_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_key_Point3D(
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
    using DataType = shopee_interfaces::msg::PackeeDetectedProduct;
    is_plain =
      (
      offsetof(DataType, position) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}


static bool _PackeeDetectedProduct__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const shopee_interfaces::msg::PackeeDetectedProduct *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _PackeeDetectedProduct__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<shopee_interfaces::msg::PackeeDetectedProduct *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _PackeeDetectedProduct__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const shopee_interfaces::msg::PackeeDetectedProduct *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _PackeeDetectedProduct__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_PackeeDetectedProduct(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _PackeeDetectedProduct__callbacks = {
  "shopee_interfaces::msg",
  "PackeeDetectedProduct",
  _PackeeDetectedProduct__cdr_serialize,
  _PackeeDetectedProduct__cdr_deserialize,
  _PackeeDetectedProduct__get_serialized_size,
  _PackeeDetectedProduct__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _PackeeDetectedProduct__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_PackeeDetectedProduct__callbacks,
  get_message_typesupport_handle_function,
  &shopee_interfaces__msg__PackeeDetectedProduct__get_type_hash,
  &shopee_interfaces__msg__PackeeDetectedProduct__get_type_description,
  &shopee_interfaces__msg__PackeeDetectedProduct__get_type_description_sources,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace shopee_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_shopee_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<shopee_interfaces::msg::PackeeDetectedProduct>()
{
  return &shopee_interfaces::msg::typesupport_fastrtps_cpp::_PackeeDetectedProduct__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, shopee_interfaces, msg, PackeeDetectedProduct)() {
  return &shopee_interfaces::msg::typesupport_fastrtps_cpp::_PackeeDetectedProduct__handle;
}

#ifdef __cplusplus
}
#endif
