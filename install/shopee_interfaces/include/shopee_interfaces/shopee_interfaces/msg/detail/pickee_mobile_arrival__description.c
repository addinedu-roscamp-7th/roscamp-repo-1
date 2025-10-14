// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeMobileArrival.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_mobile_arrival__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeMobileArrival__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x16, 0x23, 0xe6, 0x6e, 0x6d, 0x22, 0x4c, 0xca,
      0xb0, 0xc6, 0x97, 0xc5, 0xfa, 0x2a, 0x9e, 0x3a,
      0xfb, 0x65, 0x44, 0x9b, 0xf9, 0xd6, 0x2a, 0xbb,
      0x78, 0xf2, 0xe8, 0x3b, 0x29, 0xe8, 0x29, 0xc0,
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

static char shopee_interfaces__msg__PickeeMobileArrival__TYPE_NAME[] = "shopee_interfaces/msg/PickeeMobileArrival";
static char shopee_interfaces__msg__Pose2D__TYPE_NAME[] = "shopee_interfaces/msg/Pose2D";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__location_id[] = "location_id";
static char shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__final_pose[] = "final_pose";
static char shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__position_error[] = "position_error";
static char shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__travel_time[] = "travel_time";
static char shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeMobileArrival__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__location_id, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__final_pose, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__position_error, 14, 14},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__travel_time, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeMobileArrival__FIELD_NAME__message, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__msg__PickeeMobileArrival__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PickeeMobileArrival__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeMobileArrival__TYPE_NAME, 41, 41},
      {shopee_interfaces__msg__PickeeMobileArrival__FIELDS, 7, 7},
    },
    {shopee_interfaces__msg__PickeeMobileArrival__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
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
  "int32 location_id\n"
  "shopee_interfaces/Pose2D final_pose\n"
  "float32 position_error\n"
  "float32 travel_time\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeMobileArrival__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeMobileArrival__TYPE_NAME, 41, 41},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 142, 142},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeMobileArrival__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeMobileArrival__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__Pose2D__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
