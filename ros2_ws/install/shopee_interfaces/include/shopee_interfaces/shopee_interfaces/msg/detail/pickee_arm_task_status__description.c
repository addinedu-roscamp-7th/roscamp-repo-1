// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeArmTaskStatus.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_arm_task_status__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeArmTaskStatus__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x4c, 0x48, 0x30, 0xf8, 0xcc, 0x1d, 0x4b, 0x36,
      0x4c, 0x37, 0x7a, 0xed, 0xe7, 0xac, 0xb5, 0x3c,
      0xde, 0xe1, 0xa0, 0xe5, 0xe6, 0xa3, 0xb0, 0xbb,
      0x11, 0xe9, 0xea, 0x1b, 0x59, 0xd1, 0xa1, 0xc1,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PickeeArmTaskStatus__TYPE_NAME[] = "shopee_interfaces/msg/PickeeArmTaskStatus";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__product_id[] = "product_id";
static char shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__status[] = "status";
static char shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__current_phase[] = "current_phase";
static char shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__progress[] = "progress";
static char shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeArmTaskStatus__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__product_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__status, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__current_phase, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__progress, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArmTaskStatus__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__msg__PickeeArmTaskStatus__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeArmTaskStatus__TYPE_NAME, 41, 41},
      {shopee_interfaces__msg__PickeeArmTaskStatus__FIELDS, 7, 7},
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
  "int32 product_id\n"
  "string status\n"
  "string current_phase\n"
  "float32 progress\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeArmTaskStatus__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeArmTaskStatus__TYPE_NAME, 41, 41},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 114, 114},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeArmTaskStatus__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeArmTaskStatus__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
