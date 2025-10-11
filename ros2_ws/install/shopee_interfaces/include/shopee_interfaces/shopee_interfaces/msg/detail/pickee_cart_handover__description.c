// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeCartHandover.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_cart_handover__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeCartHandover__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x5f, 0x21, 0xac, 0xbf, 0xbf, 0x7d, 0x72, 0x0d,
      0x51, 0x5b, 0xd0, 0x3c, 0x74, 0x34, 0xc0, 0xac,
      0x67, 0x31, 0x6c, 0xd7, 0x64, 0x5e, 0x8d, 0x38,
      0xfd, 0x51, 0x43, 0xb2, 0xd0, 0xba, 0x3a, 0x36,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PickeeCartHandover__TYPE_NAME[] = "shopee_interfaces/msg/PickeeCartHandover";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeCartHandover__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeCartHandover__FIELD_NAME__order_id[] = "order_id";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeCartHandover__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeCartHandover__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeCartHandover__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PickeeCartHandover__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeCartHandover__TYPE_NAME, 40, 40},
      {shopee_interfaces__msg__PickeeCartHandover__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "int32 order_id";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeCartHandover__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeCartHandover__TYPE_NAME, 40, 40},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 30, 30},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeCartHandover__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeCartHandover__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
