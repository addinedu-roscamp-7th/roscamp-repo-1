// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/Pose2D.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pose2_d__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__Pose2D__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x71, 0x2f, 0x30, 0x9e, 0xb8, 0xf2, 0x39, 0x8a,
      0x73, 0x09, 0x3e, 0x1c, 0x51, 0x17, 0x2a, 0x0b,
      0xe0, 0xaf, 0x80, 0x0a, 0x54, 0x8c, 0xd7, 0x9b,
      0xe7, 0xf4, 0xbe, 0x49, 0x0e, 0xeb, 0x1d, 0x90,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__Pose2D__TYPE_NAME[] = "shopee_interfaces/msg/Pose2D";

// Define type names, field names, and default values
static char shopee_interfaces__msg__Pose2D__FIELD_NAME__x[] = "x";
static char shopee_interfaces__msg__Pose2D__FIELD_NAME__y[] = "y";
static char shopee_interfaces__msg__Pose2D__FIELD_NAME__theta[] = "theta";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__Pose2D__FIELDS[] = {
  {
    {shopee_interfaces__msg__Pose2D__FIELD_NAME__x, 1, 1},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Pose2D__FIELD_NAME__y, 1, 1},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Pose2D__FIELD_NAME__theta, 5, 5},
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
shopee_interfaces__msg__Pose2D__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
      {shopee_interfaces__msg__Pose2D__FIELDS, 3, 3},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "float32 x\n"
  "float32 y\n"
  "float32 theta";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__Pose2D__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 34, 34},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__Pose2D__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__Pose2D__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
