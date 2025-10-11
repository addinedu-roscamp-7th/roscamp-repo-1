// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeArrival.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_arrival__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeArrival__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x56, 0x63, 0xeb, 0xd9, 0xeb, 0x07, 0x7a, 0xab,
      0xfb, 0xba, 0xb9, 0x9d, 0x39, 0x40, 0xf8, 0x35,
      0x2f, 0x91, 0xd2, 0x32, 0x9a, 0xaa, 0x9f, 0x8f,
      0xf7, 0x65, 0xbd, 0x35, 0xfa, 0x82, 0x1d, 0xc8,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PickeeArrival__TYPE_NAME[] = "shopee_interfaces/msg/PickeeArrival";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeArrival__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeArrival__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__msg__PickeeArrival__FIELD_NAME__location_id[] = "location_id";
static char shopee_interfaces__msg__PickeeArrival__FIELD_NAME__section_id[] = "section_id";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeArrival__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeArrival__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArrival__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArrival__FIELD_NAME__location_id, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeArrival__FIELD_NAME__section_id, 10, 10},
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
shopee_interfaces__msg__PickeeArrival__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeArrival__TYPE_NAME, 35, 35},
      {shopee_interfaces__msg__PickeeArrival__FIELDS, 4, 4},
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
  "int32 location_id\n"
  "int32 section_id";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeArrival__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeArrival__TYPE_NAME, 35, 35},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 65, 65},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeArrival__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeArrival__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
