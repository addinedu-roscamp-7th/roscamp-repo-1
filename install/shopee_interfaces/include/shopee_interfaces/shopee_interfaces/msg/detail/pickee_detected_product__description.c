// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeDetectedProduct.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_detected_product__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeDetectedProduct__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x85, 0x64, 0x17, 0x2f, 0x9f, 0xee, 0xe7, 0x9e,
      0xef, 0x00, 0x22, 0x9a, 0x49, 0x79, 0x8e, 0x89,
      0x2c, 0xdc, 0xc4, 0xdd, 0x62, 0x09, 0xbf, 0x43,
      0x97, 0x79, 0xe1, 0x85, 0xfe, 0xc1, 0xc6, 0xfb,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "shopee_interfaces/msg/detail/b_box__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t shopee_interfaces__msg__BBox__EXPECTED_HASH = {1, {
    0x8f, 0x95, 0x00, 0x03, 0x43, 0x9e, 0x8f, 0x13,
    0xe3, 0x0e, 0x68, 0x02, 0x5f, 0xf5, 0xb4, 0xc3,
    0xc6, 0xbe, 0x4d, 0x17, 0xba, 0x2b, 0x8a, 0xe7,
    0x8a, 0xdd, 0xd2, 0xa7, 0x5e, 0x5b, 0x1f, 0xc5,
  }};
#endif

static char shopee_interfaces__msg__PickeeDetectedProduct__TYPE_NAME[] = "shopee_interfaces/msg/PickeeDetectedProduct";
static char shopee_interfaces__msg__BBox__TYPE_NAME[] = "shopee_interfaces/msg/BBox";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeDetectedProduct__FIELD_NAME__product_id[] = "product_id";
static char shopee_interfaces__msg__PickeeDetectedProduct__FIELD_NAME__bbox_number[] = "bbox_number";
static char shopee_interfaces__msg__PickeeDetectedProduct__FIELD_NAME__bbox_coords[] = "bbox_coords";
static char shopee_interfaces__msg__PickeeDetectedProduct__FIELD_NAME__confidence[] = "confidence";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeDetectedProduct__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeDetectedProduct__FIELD_NAME__product_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeDetectedProduct__FIELD_NAME__bbox_number, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeDetectedProduct__FIELD_NAME__bbox_coords, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeDetectedProduct__FIELD_NAME__confidence, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__msg__PickeeDetectedProduct__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PickeeDetectedProduct__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeDetectedProduct__TYPE_NAME, 43, 43},
      {shopee_interfaces__msg__PickeeDetectedProduct__FIELDS, 4, 4},
    },
    {shopee_interfaces__msg__PickeeDetectedProduct__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&shopee_interfaces__msg__BBox__EXPECTED_HASH, shopee_interfaces__msg__BBox__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = shopee_interfaces__msg__BBox__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 product_id\n"
  "int32 bbox_number\n"
  "shopee_interfaces/BBox bbox_coords\n"
  "float32 confidence";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeDetectedProduct__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeDetectedProduct__TYPE_NAME, 43, 43},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 89, 89},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeDetectedProduct__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeDetectedProduct__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__BBox__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
