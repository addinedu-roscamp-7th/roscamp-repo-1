// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeProductDetection.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_product_detection__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeProductDetection__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xd0, 0xa3, 0xec, 0x4d, 0xb6, 0x39, 0x21, 0xcd,
      0x38, 0x84, 0xa6, 0x0d, 0x49, 0x29, 0xe7, 0x81,
      0x76, 0xf3, 0x12, 0x78, 0x64, 0x36, 0x04, 0x79,
      0xd1, 0x67, 0x04, 0x17, 0xb5, 0xf0, 0xb5, 0x6e,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "shopee_interfaces/msg/detail/pickee_detected_product__functions.h"
#include "shopee_interfaces/msg/detail/b_box__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t shopee_interfaces__msg__BBox__EXPECTED_HASH = {1, {
    0x8f, 0x95, 0x00, 0x03, 0x43, 0x9e, 0x8f, 0x13,
    0xe3, 0x0e, 0x68, 0x02, 0x5f, 0xf5, 0xb4, 0xc3,
    0xc6, 0xbe, 0x4d, 0x17, 0xba, 0x2b, 0x8a, 0xe7,
    0x8a, 0xdd, 0xd2, 0xa7, 0x5e, 0x5b, 0x1f, 0xc5,
  }};
static const rosidl_type_hash_t shopee_interfaces__msg__PickeeDetectedProduct__EXPECTED_HASH = {1, {
    0x85, 0x64, 0x17, 0x2f, 0x9f, 0xee, 0xe7, 0x9e,
    0xef, 0x00, 0x22, 0x9a, 0x49, 0x79, 0x8e, 0x89,
    0x2c, 0xdc, 0xc4, 0xdd, 0x62, 0x09, 0xbf, 0x43,
    0x97, 0x79, 0xe1, 0x85, 0xfe, 0xc1, 0xc6, 0xfb,
  }};
#endif

static char shopee_interfaces__msg__PickeeProductDetection__TYPE_NAME[] = "shopee_interfaces/msg/PickeeProductDetection";
static char shopee_interfaces__msg__BBox__TYPE_NAME[] = "shopee_interfaces/msg/BBox";
static char shopee_interfaces__msg__PickeeDetectedProduct__TYPE_NAME[] = "shopee_interfaces/msg/PickeeDetectedProduct";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeProductDetection__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeProductDetection__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PickeeProductDetection__FIELD_NAME__products[] = "products";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeProductDetection__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeProductDetection__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeProductDetection__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeProductDetection__FIELD_NAME__products, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_UNBOUNDED_SEQUENCE,
      0,
      0,
      {shopee_interfaces__msg__PickeeDetectedProduct__TYPE_NAME, 43, 43},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__msg__PickeeProductDetection__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeDetectedProduct__TYPE_NAME, 43, 43},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PickeeProductDetection__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeProductDetection__TYPE_NAME, 44, 44},
      {shopee_interfaces__msg__PickeeProductDetection__FIELDS, 3, 3},
    },
    {shopee_interfaces__msg__PickeeProductDetection__REFERENCED_TYPE_DESCRIPTIONS, 2, 2},
  };
  if (!constructed) {
    assert(0 == memcmp(&shopee_interfaces__msg__BBox__EXPECTED_HASH, shopee_interfaces__msg__BBox__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = shopee_interfaces__msg__BBox__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__PickeeDetectedProduct__EXPECTED_HASH, shopee_interfaces__msg__PickeeDetectedProduct__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = shopee_interfaces__msg__PickeeDetectedProduct__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "int32 order_id\n"
  "shopee_interfaces/PickeeDetectedProduct[] products";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeProductDetection__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeProductDetection__TYPE_NAME, 44, 44},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 81, 81},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeProductDetection__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[3];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 3, 3};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeProductDetection__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__BBox__get_individual_type_description_source(NULL);
    sources[2] = *shopee_interfaces__msg__PickeeDetectedProduct__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
