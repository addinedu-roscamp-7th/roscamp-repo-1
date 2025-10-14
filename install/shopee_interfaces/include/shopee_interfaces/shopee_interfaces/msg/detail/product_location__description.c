// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/ProductLocation.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/product_location__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__ProductLocation__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x4f, 0xf1, 0x5a, 0xf8, 0x5e, 0x67, 0x83, 0xc9,
      0x23, 0xbc, 0xe3, 0xb1, 0xf4, 0xa5, 0x4c, 0xe8,
      0xfa, 0x16, 0x22, 0xea, 0x5f, 0x07, 0xaf, 0x66,
      0x78, 0x5a, 0x26, 0x0d, 0xf2, 0xc5, 0x7f, 0xc2,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__ProductLocation__TYPE_NAME[] = "shopee_interfaces/msg/ProductLocation";

// Define type names, field names, and default values
static char shopee_interfaces__msg__ProductLocation__FIELD_NAME__product_id[] = "product_id";
static char shopee_interfaces__msg__ProductLocation__FIELD_NAME__location_id[] = "location_id";
static char shopee_interfaces__msg__ProductLocation__FIELD_NAME__section_id[] = "section_id";
static char shopee_interfaces__msg__ProductLocation__FIELD_NAME__quantity[] = "quantity";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__ProductLocation__FIELDS[] = {
  {
    {shopee_interfaces__msg__ProductLocation__FIELD_NAME__product_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__ProductLocation__FIELD_NAME__location_id, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__ProductLocation__FIELD_NAME__section_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__ProductLocation__FIELD_NAME__quantity, 8, 8},
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
shopee_interfaces__msg__ProductLocation__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__ProductLocation__TYPE_NAME, 37, 37},
      {shopee_interfaces__msg__ProductLocation__FIELDS, 4, 4},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 product_id\n"
  "int32 location_id\n"
  "int32 section_id\n"
  "int32 quantity";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__ProductLocation__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__ProductLocation__TYPE_NAME, 37, 37},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 67, 67},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__ProductLocation__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__ProductLocation__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
