// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from shopee_interfaces:msg/BBox.idl
// generated code does not contain a copyright notice
#ifndef SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "shopee_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "shopee_interfaces/msg/detail/b_box__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_shopee_interfaces__msg__BBox(
  const shopee_interfaces__msg__BBox * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_deserialize_shopee_interfaces__msg__BBox(
  eprosima::fastcdr::Cdr &,
  shopee_interfaces__msg__BBox * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_shopee_interfaces__msg__BBox(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_shopee_interfaces__msg__BBox(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
bool cdr_serialize_key_shopee_interfaces__msg__BBox(
  const shopee_interfaces__msg__BBox * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t get_serialized_size_key_shopee_interfaces__msg__BBox(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
size_t max_serialized_size_key_shopee_interfaces__msg__BBox(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_shopee_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, shopee_interfaces, msg, BBox)();

#ifdef __cplusplus
}
#endif

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
