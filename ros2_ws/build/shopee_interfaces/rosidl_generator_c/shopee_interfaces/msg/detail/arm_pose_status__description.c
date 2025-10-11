// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/ArmPoseStatus.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/arm_pose_status__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__ArmPoseStatus__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x78, 0x3d, 0xb2, 0x54, 0x4e, 0x68, 0x8c, 0x8f,
      0xef, 0xc6, 0x5f, 0x7c, 0x05, 0xf5, 0xaa, 0x49,
      0x14, 0x97, 0x04, 0x7a, 0x99, 0x82, 0x04, 0xe3,
      0xf7, 0x82, 0xbc, 0xc2, 0x33, 0x42, 0xcb, 0x99,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__ArmPoseStatus__TYPE_NAME[] = "shopee_interfaces/msg/ArmPoseStatus";

// Define type names, field names, and default values
static char shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__pose_type[] = "pose_type";
static char shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__status[] = "status";
static char shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__progress[] = "progress";
static char shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__ArmPoseStatus__FIELDS[] = {
  {
    {shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__pose_type, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__status, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__progress, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__ArmPoseStatus__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__msg__ArmPoseStatus__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__ArmPoseStatus__TYPE_NAME, 35, 35},
      {shopee_interfaces__msg__ArmPoseStatus__FIELDS, 6, 6},
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
  "string pose_type\n"
  "string status\n"
  "float32 progress\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__ArmPoseStatus__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__ArmPoseStatus__TYPE_NAME, 35, 35},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 93, 93},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__ArmPoseStatus__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__ArmPoseStatus__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
