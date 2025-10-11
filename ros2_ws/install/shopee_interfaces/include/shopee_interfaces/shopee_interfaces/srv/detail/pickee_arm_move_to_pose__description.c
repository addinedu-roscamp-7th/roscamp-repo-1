// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:srv/PickeeArmMoveToPose.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/srv/detail/pickee_arm_move_to_pose__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeArmMoveToPose__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x75, 0x4e, 0xe0, 0x7a, 0xb3, 0xf1, 0x8a, 0x87,
      0x4a, 0x84, 0xd0, 0x7e, 0x18, 0xdc, 0x73, 0x66,
      0xb0, 0x61, 0x2b, 0xad, 0xec, 0xce, 0x2f, 0x26,
      0xba, 0xc6, 0xb6, 0x3a, 0xdb, 0x1f, 0xbe, 0x20,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xae, 0x9a, 0x12, 0xd5, 0xa5, 0x98, 0xc6, 0x67,
      0x6d, 0xf4, 0x72, 0xad, 0xc9, 0x96, 0x0b, 0x10,
      0x4b, 0xa2, 0x83, 0xcd, 0x54, 0xe7, 0x36, 0x72,
      0xa8, 0xea, 0x7c, 0xf6, 0x63, 0x94, 0xe0, 0x03,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xd8, 0x4d, 0xb3, 0x3d, 0x2e, 0x49, 0xb9, 0x1f,
      0xc4, 0x75, 0xd0, 0x1e, 0xe8, 0xdc, 0x57, 0x46,
      0x96, 0x05, 0xe9, 0x48, 0xc3, 0x36, 0x28, 0x47,
      0xd9, 0x37, 0xeb, 0x90, 0x6d, 0xcb, 0xc6, 0xbf,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeArmMoveToPose_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x8b, 0x1b, 0xd7, 0x8c, 0x49, 0xaf, 0xb2, 0xe3,
      0x6c, 0x43, 0x5d, 0x73, 0x38, 0x8b, 0x7a, 0xe9,
      0x5e, 0x0f, 0x73, 0x71, 0x7f, 0x85, 0xc1, 0xb2,
      0x8c, 0x8e, 0x4c, 0x0a, 0xd1, 0x88, 0x7e, 0xb5,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "service_msgs/msg/detail/service_event_info__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char shopee_interfaces__srv__PickeeArmMoveToPose__TYPE_NAME[] = "shopee_interfaces/srv/PickeeArmMoveToPose";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";
static char shopee_interfaces__srv__PickeeArmMoveToPose_Event__TYPE_NAME[] = "shopee_interfaces/srv/PickeeArmMoveToPose_Event";
static char shopee_interfaces__srv__PickeeArmMoveToPose_Request__TYPE_NAME[] = "shopee_interfaces/srv/PickeeArmMoveToPose_Request";
static char shopee_interfaces__srv__PickeeArmMoveToPose_Response__TYPE_NAME[] = "shopee_interfaces/srv/PickeeArmMoveToPose_Response";

// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeArmMoveToPose__FIELD_NAME__request_message[] = "request_message";
static char shopee_interfaces__srv__PickeeArmMoveToPose__FIELD_NAME__response_message[] = "response_message";
static char shopee_interfaces__srv__PickeeArmMoveToPose__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeArmMoveToPose__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeArmMoveToPose_Request__TYPE_NAME, 49, 49},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeArmMoveToPose_Response__TYPE_NAME, 50, 50},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeArmMoveToPose_Event__TYPE_NAME, 47, 47},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PickeeArmMoveToPose__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Event__TYPE_NAME, 47, 47},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Request__TYPE_NAME, 49, 49},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Response__TYPE_NAME, 50, 50},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PickeeArmMoveToPose__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeArmMoveToPose__TYPE_NAME, 41, 41},
      {shopee_interfaces__srv__PickeeArmMoveToPose__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PickeeArmMoveToPose__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__srv__PickeeArmMoveToPose_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeArmMoveToPose_Request__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__srv__PickeeArmMoveToPose_Request__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__srv__PickeeArmMoveToPose_Request__FIELD_NAME__pose_type[] = "pose_type";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeArmMoveToPose_Request__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Request__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Request__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Request__FIELD_NAME__pose_type, 9, 9},
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
shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeArmMoveToPose_Request__TYPE_NAME, 49, 49},
      {shopee_interfaces__srv__PickeeArmMoveToPose_Request__FIELDS, 3, 3},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeArmMoveToPose_Response__FIELD_NAME__success[] = "success";
static char shopee_interfaces__srv__PickeeArmMoveToPose_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeArmMoveToPose_Response__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Response__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeArmMoveToPose_Response__TYPE_NAME, 50, 50},
      {shopee_interfaces__srv__PickeeArmMoveToPose_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeArmMoveToPose_Event__FIELD_NAME__info[] = "info";
static char shopee_interfaces__srv__PickeeArmMoveToPose_Event__FIELD_NAME__request[] = "request";
static char shopee_interfaces__srv__PickeeArmMoveToPose_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeArmMoveToPose_Event__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PickeeArmMoveToPose_Request__TYPE_NAME, 49, 49},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PickeeArmMoveToPose_Response__TYPE_NAME, 50, 50},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PickeeArmMoveToPose_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Request__TYPE_NAME, 49, 49},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Response__TYPE_NAME, 50, 50},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PickeeArmMoveToPose_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeArmMoveToPose_Event__TYPE_NAME, 47, 47},
      {shopee_interfaces__srv__PickeeArmMoveToPose_Event__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PickeeArmMoveToPose_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "int32 order_id\n"
  "string pose_type\n"
  "---\n"
  "bool success\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeArmMoveToPose__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeArmMoveToPose__TYPE_NAME, 41, 41},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 79, 79},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Request__TYPE_NAME, 49, 49},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Response__TYPE_NAME, 50, 50},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeArmMoveToPose_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeArmMoveToPose_Event__TYPE_NAME, 47, 47},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeArmMoveToPose__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeArmMoveToPose__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__srv__PickeeArmMoveToPose_Event__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_individual_type_description_source(NULL);
    sources[5] = *shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeArmMoveToPose_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeArmMoveToPose_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__srv__PickeeArmMoveToPose_Request__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__srv__PickeeArmMoveToPose_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
