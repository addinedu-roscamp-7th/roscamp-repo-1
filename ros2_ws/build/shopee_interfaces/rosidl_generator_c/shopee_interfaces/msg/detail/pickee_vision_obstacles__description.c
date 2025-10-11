// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeVisionObstacles.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_vision_obstacles__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeVisionObstacles__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x82, 0x94, 0xea, 0x43, 0x71, 0x28, 0xb2, 0xe0,
      0x17, 0xc4, 0x60, 0x0b, 0xf7, 0xf6, 0xc8, 0x56,
      0x6f, 0xb4, 0x9b, 0x87, 0x78, 0x68, 0xdb, 0x6c,
      0x9b, 0xcc, 0x30, 0x98, 0xc2, 0xd1, 0x3d, 0xed,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "shopee_interfaces/msg/detail/point2_d__functions.h"
#include "shopee_interfaces/msg/detail/vector2_d__functions.h"
#include "shopee_interfaces/msg/detail/obstacle__functions.h"
#include "shopee_interfaces/msg/detail/b_box__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t shopee_interfaces__msg__BBox__EXPECTED_HASH = {1, {
    0x8f, 0x95, 0x00, 0x03, 0x43, 0x9e, 0x8f, 0x13,
    0xe3, 0x0e, 0x68, 0x02, 0x5f, 0xf5, 0xb4, 0xc3,
    0xc6, 0xbe, 0x4d, 0x17, 0xba, 0x2b, 0x8a, 0xe7,
    0x8a, 0xdd, 0xd2, 0xa7, 0x5e, 0x5b, 0x1f, 0xc5,
  }};
static const rosidl_type_hash_t shopee_interfaces__msg__Obstacle__EXPECTED_HASH = {1, {
    0xbc, 0x17, 0xa8, 0x75, 0x38, 0x95, 0xdf, 0x9f,
    0xaa, 0xbb, 0x14, 0xb1, 0x47, 0x46, 0x51, 0xa4,
    0x95, 0xe3, 0x37, 0x74, 0xc1, 0x39, 0x01, 0xb1,
    0xe2, 0x94, 0xb3, 0x0f, 0xec, 0x4f, 0xd1, 0xe2,
  }};
static const rosidl_type_hash_t shopee_interfaces__msg__Point2D__EXPECTED_HASH = {1, {
    0x9c, 0xfd, 0xe0, 0xd2, 0xef, 0x71, 0x92, 0xcc,
    0x15, 0xa2, 0xe5, 0x63, 0xae, 0xf9, 0x17, 0xe6,
    0x0d, 0x63, 0x27, 0x76, 0x38, 0x3c, 0x6a, 0x1f,
    0xf4, 0x36, 0xed, 0x05, 0x97, 0xaa, 0x9c, 0x0d,
  }};
static const rosidl_type_hash_t shopee_interfaces__msg__Vector2D__EXPECTED_HASH = {1, {
    0x77, 0x99, 0x64, 0xe1, 0x6d, 0x03, 0x50, 0x0e,
    0x8e, 0x3b, 0x1c, 0xaa, 0xca, 0x2d, 0x3b, 0x4f,
    0x63, 0x2f, 0xa8, 0x3c, 0x57, 0xc2, 0x04, 0xc5,
    0x0e, 0x8f, 0xf0, 0x17, 0x3e, 0x67, 0x09, 0x2d,
  }};
#endif

static char shopee_interfaces__msg__PickeeVisionObstacles__TYPE_NAME[] = "shopee_interfaces/msg/PickeeVisionObstacles";
static char shopee_interfaces__msg__BBox__TYPE_NAME[] = "shopee_interfaces/msg/BBox";
static char shopee_interfaces__msg__Obstacle__TYPE_NAME[] = "shopee_interfaces/msg/Obstacle";
static char shopee_interfaces__msg__Point2D__TYPE_NAME[] = "shopee_interfaces/msg/Point2D";
static char shopee_interfaces__msg__Vector2D__TYPE_NAME[] = "shopee_interfaces/msg/Vector2D";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeVisionObstacles__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeVisionObstacles__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PickeeVisionObstacles__FIELD_NAME__obstacles[] = "obstacles";
static char shopee_interfaces__msg__PickeeVisionObstacles__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeVisionObstacles__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeVisionObstacles__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionObstacles__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionObstacles__FIELD_NAME__obstacles, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_UNBOUNDED_SEQUENCE,
      0,
      0,
      {shopee_interfaces__msg__Obstacle__TYPE_NAME, 30, 30},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionObstacles__FIELD_NAME__message, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__msg__PickeeVisionObstacles__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Obstacle__TYPE_NAME, 30, 30},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Point2D__TYPE_NAME, 29, 29},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Vector2D__TYPE_NAME, 30, 30},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PickeeVisionObstacles__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeVisionObstacles__TYPE_NAME, 43, 43},
      {shopee_interfaces__msg__PickeeVisionObstacles__FIELDS, 4, 4},
    },
    {shopee_interfaces__msg__PickeeVisionObstacles__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&shopee_interfaces__msg__BBox__EXPECTED_HASH, shopee_interfaces__msg__BBox__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = shopee_interfaces__msg__BBox__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__Obstacle__EXPECTED_HASH, shopee_interfaces__msg__Obstacle__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = shopee_interfaces__msg__Obstacle__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__Point2D__EXPECTED_HASH, shopee_interfaces__msg__Point2D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__msg__Point2D__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__Vector2D__EXPECTED_HASH, shopee_interfaces__msg__Vector2D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__msg__Vector2D__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "int32 order_id\n"
  "shopee_interfaces/Obstacle[] obstacles\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeVisionObstacles__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeVisionObstacles__TYPE_NAME, 43, 43},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 84, 84},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeVisionObstacles__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeVisionObstacles__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__BBox__get_individual_type_description_source(NULL);
    sources[2] = *shopee_interfaces__msg__Obstacle__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__msg__Point2D__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__msg__Vector2D__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
