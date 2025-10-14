// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeMobilePose.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_mobile_pose__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeMobilePose__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x5e, 0x6a, 0x5e, 0xd8, 0x0f, 0xe8, 0x22, 0x58,
      0x15, 0x66, 0xd8, 0x75, 0xe1, 0x44, 0x02, 0xc5,
      0xce, 0xc7, 0x0a, 0xc1, 0x65, 0x7e, 0x4a, 0x53,
      0x25, 0x2e, 0x26, 0x26, 0xc1, 0xb4, 0x19, 0xcb,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "shopee_interfaces/msg/detail/pose2_d__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t shopee_interfaces__msg__Pose2D__EXPECTED_HASH = {1, {
    0x71, 0x2f, 0x30, 0x9e, 0xb8, 0xf2, 0x39, 0x8a,
    0x73, 0x09, 0x3e, 0x1c, 0x51, 0x17, 0x2a, 0x0b,
    0xe0, 0xaf, 0x80, 0x0a, 0x54, 0x8c, 0xd7, 0x9b,
    0xe7, 0xf4, 0xbe, 0x49, 0x0e, 0xeb, 0x1d, 0x90,
  }};
#endif

static char shopee_interfaces__msg__PickeeMobilePose__TYPE_NAME[] = "shopee_interfaces/msg/PickeeMobilePose";
static char shopee_interfaces__msg__Pose2D__TYPE_NAME[] = "shopee_interfaces/msg/Pose2D";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__current_pose[] = "current_pose";
static char shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__linear_velocity[] = "linear_velocity";
static char shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__angular_velocity[] = "angular_velocity";
static char shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__battery_level[] = "battery_level";
static char shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__status[] = "status";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeMobilePose__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__current_pose, 12, 12},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__linear_velocity, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__angular_velocity, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__battery_level, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobilePose__FIELD_NAME__status, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__msg__PickeeMobilePose__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PickeeMobilePose__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeMobilePose__TYPE_NAME, 38, 38},
      {shopee_interfaces__msg__PickeeMobilePose__FIELDS, 7, 7},
    },
    {shopee_interfaces__msg__PickeeMobilePose__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&shopee_interfaces__msg__Pose2D__EXPECTED_HASH, shopee_interfaces__msg__Pose2D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = shopee_interfaces__msg__Pose2D__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "int32 order_id\n"
  "shopee_interfaces/Pose2D current_pose\n"
  "float32 linear_velocity\n"
  "float32 angular_velocity\n"
  "float32 battery_level\n"
  "string status";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeMobilePose__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeMobilePose__TYPE_NAME, 38, 38},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 153, 153},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeMobilePose__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeMobilePose__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__Pose2D__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
