// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/Point3D.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/point3_d__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__Point3D__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x5d, 0xfb, 0x73, 0x9e, 0xcd, 0x7c, 0xb8, 0x03,
      0x2d, 0xad, 0x5e, 0xde, 0xda, 0xb9, 0x23, 0xb5,
      0x00, 0xb3, 0x13, 0x60, 0x8e, 0x2a, 0x71, 0x53,
      0x92, 0x08, 0xaf, 0xf2, 0x91, 0x28, 0x6f, 0xa2,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__Point3D__TYPE_NAME[] = "shopee_interfaces/msg/Point3D";

// Define type names, field names, and default values
static char shopee_interfaces__msg__Point3D__FIELD_NAME__x[] = "x";
static char shopee_interfaces__msg__Point3D__FIELD_NAME__y[] = "y";
static char shopee_interfaces__msg__Point3D__FIELD_NAME__z[] = "z";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__Point3D__FIELDS[] = {
  {
    {shopee_interfaces__msg__Point3D__FIELD_NAME__x, 1, 1},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Point3D__FIELD_NAME__y, 1, 1},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Point3D__FIELD_NAME__z, 1, 1},
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
shopee_interfaces__msg__Point3D__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__Point3D__TYPE_NAME, 29, 29},
      {shopee_interfaces__msg__Point3D__FIELDS, 3, 3},
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
  "float32 z";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__Point3D__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__Point3D__TYPE_NAME, 29, 29},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 30, 30},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__Point3D__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__Point3D__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
