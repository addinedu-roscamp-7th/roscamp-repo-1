// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeVisionCartCheck.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_vision_cart_check__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeVisionCartCheck__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x8f, 0xf9, 0x07, 0xfa, 0x65, 0xdd, 0xc8, 0xad,
      0xcd, 0xf7, 0x01, 0x36, 0xa7, 0x03, 0x77, 0x57,
      0x9a, 0x7a, 0x09, 0xe2, 0x28, 0x64, 0xcd, 0x50,
      0xfc, 0x20, 0x0c, 0x85, 0x90, 0x3d, 0xe3, 0xac,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PickeeVisionCartCheck__TYPE_NAME[] = "shopee_interfaces/msg/PickeeVisionCartCheck";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__success[] = "success";
static char shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__product_id[] = "product_id";
static char shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__found[] = "found";
static char shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__quantity[] = "quantity";
static char shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeVisionCartCheck__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__product_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__found, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__quantity, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionCartCheck__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__msg__PickeeVisionCartCheck__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeVisionCartCheck__TYPE_NAME, 43, 43},
      {shopee_interfaces__msg__PickeeVisionCartCheck__FIELDS, 7, 7},
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
  "int32 product_id\n"
  "bool found\n"
  "int32 quantity\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeVisionCartCheck__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeVisionCartCheck__TYPE_NAME, 43, 43},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 101, 101},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeVisionCartCheck__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeVisionCartCheck__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
