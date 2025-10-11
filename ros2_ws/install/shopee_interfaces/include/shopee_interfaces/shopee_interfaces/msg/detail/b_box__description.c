// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/BBox.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/b_box__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__BBox__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x8f, 0x95, 0x00, 0x03, 0x43, 0x9e, 0x8f, 0x13,
      0xe3, 0x0e, 0x68, 0x02, 0x5f, 0xf5, 0xb4, 0xc3,
      0xc6, 0xbe, 0x4d, 0x17, 0xba, 0x2b, 0x8a, 0xe7,
      0x8a, 0xdd, 0xd2, 0xa7, 0x5e, 0x5b, 0x1f, 0xc5,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__BBox__TYPE_NAME[] = "shopee_interfaces/msg/BBox";

// Define type names, field names, and default values
static char shopee_interfaces__msg__BBox__FIELD_NAME__x1[] = "x1";
static char shopee_interfaces__msg__BBox__FIELD_NAME__y1[] = "y1";
static char shopee_interfaces__msg__BBox__FIELD_NAME__x2[] = "x2";
static char shopee_interfaces__msg__BBox__FIELD_NAME__y2[] = "y2";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__BBox__FIELDS[] = {
  {
    {shopee_interfaces__msg__BBox__FIELD_NAME__x1, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__BBox__FIELD_NAME__y1, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__BBox__FIELD_NAME__x2, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__BBox__FIELD_NAME__y2, 2, 2},
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
shopee_interfaces__msg__BBox__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
      {shopee_interfaces__msg__BBox__FIELDS, 4, 4},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 x1\n"
  "int32 y1\n"
  "int32 x2\n"
  "int32 y2";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__BBox__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 36, 36},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__BBox__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__BBox__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
