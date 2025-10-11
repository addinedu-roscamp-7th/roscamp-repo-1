// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:msg/PickeeVisionStaffRegister.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/msg/detail/pickee_vision_staff_register__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__msg__PickeeVisionStaffRegister__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xdc, 0xcc, 0xab, 0x23, 0x18, 0xfb, 0x51, 0xde,
      0x48, 0xd8, 0x72, 0xe0, 0xac, 0x12, 0xaf, 0xe6,
      0xcf, 0x88, 0x61, 0x41, 0x60, 0x03, 0xe9, 0x38,
      0x5f, 0xea, 0xa0, 0xae, 0xa1, 0xe9, 0x51, 0x2c,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char shopee_interfaces__msg__PickeeVisionStaffRegister__TYPE_NAME[] = "shopee_interfaces/msg/PickeeVisionStaffRegister";

// Define type names, field names, and default values
static char shopee_interfaces__msg__PickeeVisionStaffRegister__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__msg__PickeeVisionStaffRegister__FIELD_NAME__success[] = "success";
static char shopee_interfaces__msg__PickeeVisionStaffRegister__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__msg__PickeeVisionStaffRegister__FIELDS[] = {
  {
    {shopee_interfaces__msg__PickeeVisionStaffRegister__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionStaffRegister__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__PickeeVisionStaffRegister__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__msg__PickeeVisionStaffRegister__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__msg__PickeeVisionStaffRegister__TYPE_NAME, 47, 47},
      {shopee_interfaces__msg__PickeeVisionStaffRegister__FIELDS, 3, 3},
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
  "bool success\n"
  "string message";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__msg__PickeeVisionStaffRegister__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__msg__PickeeVisionStaffRegister__TYPE_NAME, 47, 47},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 43, 43},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__msg__PickeeVisionStaffRegister__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__msg__PickeeVisionStaffRegister__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
