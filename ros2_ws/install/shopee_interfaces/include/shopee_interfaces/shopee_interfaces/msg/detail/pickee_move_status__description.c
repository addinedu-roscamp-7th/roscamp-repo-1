// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeMoveStatus.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_move_status__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeMoveStatus__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x48, 0x8b, 0x45, 0x84, 0x86, 0x00, 0xaa, 0xc3,
      0x13, 0x34, 0x5d, 0x8d, 0x32, 0xe2, 0x0e, 0x53,
      0xb8, 0x85, 0x7a, 0x11, 0x9e, 0x97, 0x5b, 0xe6,
      0xa9, 0x80, 0xa7, 0x07, 0x0d, 0xfd, 0x6a, 0x6a,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PickeeMoveStatus__TYPE_NAME[] = "shopee_interfaces/msg/PickeeMoveStatus";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeMoveStatus__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeMoveStatus__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PickeeMoveStatus__FIELD_NAME__location_id[] = "location_id";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeMoveStatus__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeMoveStatus__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMoveStatus__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMoveStatus__FIELD_NAME__location_id, 11, 11},
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
shopee_interfaces__msg__PickeeMoveStatus__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeMoveStatus__TYPE_NAME, 38, 38},
      {shopee_interfaces__msg__PickeeMoveStatus__FIELDS, 3, 3},
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
  "int32 order_id\n"
  "int32 location_id";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeMoveStatus__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeMoveStatus__TYPE_NAME, 38, 38},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 48, 48},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeMoveStatus__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeMoveStatus__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
