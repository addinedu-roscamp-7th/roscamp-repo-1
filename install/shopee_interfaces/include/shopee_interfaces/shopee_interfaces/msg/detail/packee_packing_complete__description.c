// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PackeePackingComplete.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/packee_packing_complete__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PackeePackingComplete__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xec, 0xb6, 0x08, 0xe8, 0x07, 0x55, 0x11, 0xda,
      0x52, 0xab, 0xc2, 0x78, 0x81, 0x32, 0xe1, 0x21,
      0xf8, 0x33, 0x0c, 0x15, 0x1b, 0xf0, 0x85, 0x07,
      0x07, 0x3b, 0x0c, 0x99, 0x14, 0xcc, 0xa3, 0xea,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PackeePackingComplete__TYPE_NAME[] = "shopee_interfaces/msg/PackeePackingComplete";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__success[] = "success";
static char shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__packed_items[] = "packed_items";
static char shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PackeePackingComplete__FIELDS[] = {
  {
    {shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__packed_items, 12, 12},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeePackingComplete__FIELD_NAME__message, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PackeePackingComplete__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PackeePackingComplete__TYPE_NAME, 43, 43},
      {shopee_interfaces__msg__PackeePackingComplete__FIELDS, 5, 5},
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
  "bool success\n"
  "int32 packed_items\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PackeePackingComplete__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PackeePackingComplete__TYPE_NAME, 43, 43},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 77, 77},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PackeePackingComplete__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PackeePackingComplete__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
