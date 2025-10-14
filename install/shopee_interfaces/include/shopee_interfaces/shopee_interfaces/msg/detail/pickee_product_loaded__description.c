// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeProductLoaded.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_product_loaded__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeProductLoaded__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xc9, 0x80, 0xc5, 0x7d, 0xfe, 0x13, 0x9f, 0xbe,
      0x7e, 0x37, 0x1a, 0x24, 0xae, 0xf6, 0x89, 0xb2,
      0xd5, 0x3c, 0x1b, 0xbd, 0xa7, 0xdf, 0xed, 0xb1,
      0xa3, 0x86, 0x65, 0x5c, 0x49, 0x2a, 0xdf, 0xa8,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PickeeProductLoaded__TYPE_NAME[] = "shopee_interfaces/msg/PickeeProductLoaded";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__product_id[] = "product_id";
static char shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__quantity[] = "quantity";
static char shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__success[] = "success";
static char shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeProductLoaded__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__product_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__quantity, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeProductLoaded__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__msg__PickeeProductLoaded__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeProductLoaded__TYPE_NAME, 41, 41},
      {shopee_interfaces__msg__PickeeProductLoaded__FIELDS, 5, 5},
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
  "int32 product_id\n"
  "int32 quantity\n"
  "bool success\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeProductLoaded__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeProductLoaded__TYPE_NAME, 41, 41},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 74, 74},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeProductLoaded__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeProductLoaded__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
