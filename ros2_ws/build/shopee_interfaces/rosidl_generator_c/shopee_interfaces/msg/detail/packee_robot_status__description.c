// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PackeeRobotStatus.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/packee_robot_status__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PackeeRobotStatus__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x76, 0x6f, 0x28, 0x2f, 0xb5, 0x6e, 0x52, 0x6c,
      0x79, 0xb3, 0x99, 0x96, 0x13, 0xeb, 0x74, 0x6c,
      0x9c, 0x5f, 0x5d, 0xca, 0xce, 0xb6, 0xfe, 0x5c,
      0xe1, 0x2a, 0xa3, 0xf7, 0x0b, 0x81, 0x99, 0xcf,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PackeeRobotStatus__TYPE_NAME[] = "shopee_interfaces/msg/PackeeRobotStatus";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PackeeRobotStatus__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PackeeRobotStatus__FIELD_NAME__state[] = "state";
static char shopee_interfaces__msg__PackeeRobotStatus__FIELD_NAME__current_order_id[] = "current_order_id";
static char shopee_interfaces__msg__PackeeRobotStatus__FIELD_NAME__items_in_cart[] = "items_in_cart";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PackeeRobotStatus__FIELDS[] = {
  {
    {shopee_interfaces__msg__PackeeRobotStatus__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeRobotStatus__FIELD_NAME__state, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeRobotStatus__FIELD_NAME__current_order_id, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeRobotStatus__FIELD_NAME__items_in_cart, 13, 13},
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
shopee_interfaces__msg__PackeeRobotStatus__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PackeeRobotStatus__TYPE_NAME, 39, 39},
      {shopee_interfaces__msg__PackeeRobotStatus__FIELDS, 4, 4},
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
  "string state\n"
  "int32 current_order_id\n"
  "int32 items_in_cart";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PackeeRobotStatus__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PackeeRobotStatus__TYPE_NAME, 39, 39},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 71, 71},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PackeeRobotStatus__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PackeeRobotStatus__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
