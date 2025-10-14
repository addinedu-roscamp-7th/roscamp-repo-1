// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/Vector2D.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/vector2_d__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__Vector2D__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x77, 0x99, 0x64, 0xe1, 0x6d, 0x03, 0x50, 0x0e,
      0x8e, 0x3b, 0x1c, 0xaa, 0xca, 0x2d, 0x3b, 0x4f,
      0x63, 0x2f, 0xa8, 0x3c, 0x57, 0xc2, 0x04, 0xc5,
      0x0e, 0x8f, 0xf0, 0x17, 0x3e, 0x67, 0x09, 0x2d,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__Vector2D__TYPE_NAME[] = "shopee_interfaces/msg/Vector2D";

// Define type names, field names, and default values
static char shopee_interfaces__msg__Vector2D__FIELD_NAME__vx[] = "vx";
static char shopee_interfaces__msg__Vector2D__FIELD_NAME__vy[] = "vy";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__Vector2D__FIELDS[] = {
  {
    {shopee_interfaces__msg__Vector2D__FIELD_NAME__vx, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Vector2D__FIELD_NAME__vy, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__Vector2D__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__Vector2D__TYPE_NAME, 30, 30},
      {shopee_interfaces__msg__Vector2D__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "float32 vx\n"
  "float32 vy";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__Vector2D__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__Vector2D__TYPE_NAME, 30, 30},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 22, 22},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__Vector2D__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__Vector2D__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
