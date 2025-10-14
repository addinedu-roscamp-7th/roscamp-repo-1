// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PackeeDetectedProduct.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/packee_detected_product__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PackeeDetectedProduct__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x4c, 0xcc, 0x0d, 0x5c, 0x2e, 0x93, 0x46, 0xd4,
      0x3a, 0x7e, 0xa9, 0x4c, 0xef, 0x01, 0xd9, 0x4d,
      0x86, 0x5a, 0xf6, 0x83, 0x4b, 0x08, 0x86, 0x9c,
      0x74, 0x38, 0x97, 0xd1, 0x39, 0x3b, 0x7d, 0x73,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "shopee_interfaces/msg/detail/b_box__functions.h"
#include "shopee_interfaces/msg/detail/point3_d__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t shopee_interfaces__msg__BBox__EXPECTED_HASH = {1, {
    0x8f, 0x95, 0x00, 0x03, 0x43, 0x9e, 0x8f, 0x13,
    0xe3, 0x0e, 0x68, 0x02, 0x5f, 0xf5, 0xb4, 0xc3,
    0xc6, 0xbe, 0x4d, 0x17, 0xba, 0x2b, 0x8a, 0xe7,
    0x8a, 0xdd, 0xd2, 0xa7, 0x5e, 0x5b, 0x1f, 0xc5,
  }};
static const rosidl_type_hash_t shopee_interfaces__msg__Point3D__EXPECTED_HASH = {1, {
    0x5d, 0xfb, 0x73, 0x9e, 0xcd, 0x7c, 0xb8, 0x03,
    0x2d, 0xad, 0x5e, 0xde, 0xda, 0xb9, 0x23, 0xb5,
    0x00, 0xb3, 0x13, 0x60, 0x8e, 0x2a, 0x71, 0x53,
    0x92, 0x08, 0xaf, 0xf2, 0x91, 0x28, 0x6f, 0xa2,
  }};
#endif

static char shopee_interfaces__msg__PackeeDetectedProduct__TYPE_NAME[] = "shopee_interfaces/msg/PackeeDetectedProduct";
static char shopee_interfaces__msg__BBox__TYPE_NAME[] = "shopee_interfaces/msg/BBox";
static char shopee_interfaces__msg__Point3D__TYPE_NAME[] = "shopee_interfaces/msg/Point3D";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PackeeDetectedProduct__FIELD_NAME__product_id[] = "product_id";
static char shopee_interfaces__msg__PackeeDetectedProduct__FIELD_NAME__bbox[] = "bbox";
static char shopee_interfaces__msg__PackeeDetectedProduct__FIELD_NAME__confidence[] = "confidence";
static char shopee_interfaces__msg__PackeeDetectedProduct__FIELD_NAME__position[] = "position";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PackeeDetectedProduct__FIELDS[] = {
  {
    {shopee_interfaces__msg__PackeeDetectedProduct__FIELD_NAME__product_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeDetectedProduct__FIELD_NAME__bbox, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeDetectedProduct__FIELD_NAME__confidence, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PackeeDetectedProduct__FIELD_NAME__position, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__Point3D__TYPE_NAME, 29, 29},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__msg__PackeeDetectedProduct__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Point3D__TYPE_NAME, 29, 29},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__msg__PackeeDetectedProduct__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PackeeDetectedProduct__TYPE_NAME, 43, 43},
      {shopee_interfaces__msg__PackeeDetectedProduct__FIELDS, 4, 4},
    },
    {shopee_interfaces__msg__PackeeDetectedProduct__REFERENCED_TYPE_DESCRIPTIONS, 2, 2},
  };
  if (!constructed) {
    assert(0 == memcmp(&shopee_interfaces__msg__BBox__EXPECTED_HASH, shopee_interfaces__msg__BBox__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = shopee_interfaces__msg__BBox__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__Point3D__EXPECTED_HASH, shopee_interfaces__msg__Point3D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = shopee_interfaces__msg__Point3D__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 product_id\n"
  "shopee_interfaces/BBox bbox\n"
  "float32 confidence\n"
  "shopee_interfaces/Point3D position";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PackeeDetectedProduct__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PackeeDetectedProduct__TYPE_NAME, 43, 43},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 99, 99},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PackeeDetectedProduct__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[3];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 3, 3};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PackeeDetectedProduct__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__BBox__get_individual_type_description_source(NULL);
    sources[2] = *shopee_interfaces__msg__Point3D__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
