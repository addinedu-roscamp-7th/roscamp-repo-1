// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeVisionStaffLocation.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_vision_staff_location__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeVisionStaffLocation__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xac, 0xec, 0x6a, 0x1b, 0xba, 0x6a, 0xa5, 0xf2,
      0x4a, 0x94, 0xd0, 0x61, 0x8f, 0x9d, 0x90, 0x33,
      0xa6, 0x9b, 0x92, 0xf8, 0x7c, 0xb7, 0xa2, 0x42,
      0xf1, 0xa6, 0x6d, 0x62, 0xe4, 0xf9, 0xcc, 0x04,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "shopee_interfaces/msg/detail/point2_d__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t shopee_interfaces__msg__Point2D__EXPECTED_HASH = {1, {
    0x9c, 0xfd, 0xe0, 0xd2, 0xef, 0x71, 0x92, 0xcc,
    0x15, 0xa2, 0xe5, 0x63, 0xae, 0xf9, 0x17, 0xe6,
    0x0d, 0x63, 0x27, 0x76, 0x38, 0x3c, 0x6a, 0x1f,
    0xf4, 0x36, 0xed, 0x05, 0x97, 0xaa, 0x9c, 0x0d,
  }};
#endif

static char shopee_interfaces__msg__PickeeVisionStaffLocation__TYPE_NAME[] = "shopee_interfaces/msg/PickeeVisionStaffLocation";
static char shopee_interfaces__msg__Point2D__TYPE_NAME[] = "shopee_interfaces/msg/Point2D";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeVisionStaffLocation__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeVisionStaffLocation__FIELD_NAME__relative_position[] = "relative_position";
static char shopee_interfaces__msg__PickeeVisionStaffLocation__FIELD_NAME__distance[] = "distance";
static char shopee_interfaces__msg__PickeeVisionStaffLocation__FIELD_NAME__is_tracking[] = "is_tracking";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeVisionStaffLocation__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeVisionStaffLocation__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionStaffLocation__FIELD_NAME__relative_position, 17, 17},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__Point2D__TYPE_NAME, 29, 29},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionStaffLocation__FIELD_NAME__distance, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionStaffLocation__FIELD_NAME__is_tracking, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__msg__PickeeVisionStaffLocation__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {shopee_interfaces__msg__Point2D__TYPE_NAME, 29, 29},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PickeeVisionStaffLocation__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeVisionStaffLocation__TYPE_NAME, 47, 47},
      {shopee_interfaces__msg__PickeeVisionStaffLocation__FIELDS, 4, 4},
    },
    {shopee_interfaces__msg__PickeeVisionStaffLocation__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&shopee_interfaces__msg__Point2D__EXPECTED_HASH, shopee_interfaces__msg__Point2D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = shopee_interfaces__msg__Point2D__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "shopee_interfaces/Point2D relative_position\n"
  "float32 distance\n"
  "bool is_tracking";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeVisionStaffLocation__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeVisionStaffLocation__TYPE_NAME, 47, 47},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 93, 93},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeVisionStaffLocation__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeVisionStaffLocation__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__Point2D__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
