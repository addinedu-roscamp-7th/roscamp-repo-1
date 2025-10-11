// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PackeeArmTaskStatus.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/packee_arm_task_status__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PackeeArmTaskStatus__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x54, 0x7d, 0xf3, 0x27, 0xa8, 0x03, 0xba, 0x71,
      0xd7, 0xf0, 0xa0, 0x28, 0xdf, 0x08, 0xaf, 0x89,
      0xde, 0x09, 0xbe, 0x0f, 0x30, 0x05, 0xdb, 0x0b,
      0xd3, 0x8b, 0x5e, 0x9e, 0x18, 0x86, 0xc1, 0x8f,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PackeeArmTaskStatus__TYPE_NAME[] = "shopee_interfaces/msg/PackeeArmTaskStatus";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__product_id[] = "product_id";
static char shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__arm_side[] = "arm_side";
static char shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__status[] = "status";
static char shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__current_phase[] = "current_phase";
static char shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__progress[] = "progress";
static char shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PackeeArmTaskStatus__FIELDS[] = {
  {
    {shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__product_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__arm_side, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__status, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__current_phase, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__progress, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeArmTaskStatus__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__msg__PackeeArmTaskStatus__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PackeeArmTaskStatus__TYPE_NAME, 41, 41},
      {shopee_interfaces__msg__PackeeArmTaskStatus__FIELDS, 8, 8},
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
  "string arm_side\n"
  "string status\n"
  "string current_phase\n"
  "float32 progress\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PackeeArmTaskStatus__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PackeeArmTaskStatus__TYPE_NAME, 41, 41},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 130, 130},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PackeeArmTaskStatus__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PackeeArmTaskStatus__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
