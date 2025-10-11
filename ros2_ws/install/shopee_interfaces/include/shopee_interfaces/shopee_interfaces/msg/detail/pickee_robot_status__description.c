// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeRobotStatus.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_robot_status__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeRobotStatus__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x31, 0x50, 0x5e, 0xa9, 0xf0, 0xaa, 0xf0, 0x8e,
      0x95, 0x0b, 0xb1, 0x07, 0x0c, 0x1b, 0x10, 0xbf,
      0x25, 0x91, 0xd4, 0x6b, 0xa0, 0xfc, 0x85, 0x68,
      0x6a, 0xd0, 0x14, 0x00, 0xc4, 0xdc, 0xc3, 0x66,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PickeeRobotStatus__TYPE_NAME[] = "shopee_interfaces/msg/PickeeRobotStatus";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__state[] = "state";
static char shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__battery_level[] = "battery_level";
static char shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__current_order_id[] = "current_order_id";
static char shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__position_x[] = "position_x";
static char shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__position_y[] = "position_y";
static char shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__orientation_z[] = "orientation_z";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeRobotStatus__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__state, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__battery_level, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__current_order_id, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__position_x, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__position_y, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeRobotStatus__FIELD_NAME__orientation_z, 13, 13},
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
shopee_interfaces__msg__PickeeRobotStatus__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeRobotStatus__TYPE_NAME, 39, 39},
      {shopee_interfaces__msg__PickeeRobotStatus__FIELDS, 7, 7},
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
  "float32 battery_level\n"
  "int32 current_order_id\n"
  "float32 position_x\n"
  "float32 position_y\n"
  "float32 orientation_z";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeRobotStatus__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeRobotStatus__TYPE_NAME, 39, 39},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 133, 133},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeRobotStatus__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeRobotStatus__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
