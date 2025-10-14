// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:srv/PickeeMobileMoveToLocation.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/srv/detail/pickee_mobile_move_to_location__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeMobileMoveToLocation__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xfc, 0x59, 0x7a, 0xcf, 0x73, 0x11, 0xc8, 0xc4,
      0xb1, 0x3b, 0x86, 0xb5, 0x9e, 0x26, 0x2c, 0xa0,
      0xf7, 0xe2, 0xeb, 0x4c, 0x7c, 0x4c, 0x3e, 0xcf,
      0x3a, 0x4c, 0xd8, 0x78, 0x85, 0x65, 0x10, 0x11,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xd8, 0x5c, 0xee, 0xe8, 0x9e, 0xe6, 0x68, 0x88,
      0x4d, 0xdd, 0x33, 0xfb, 0xf2, 0x06, 0x8b, 0x87,
      0xc1, 0x46, 0x88, 0x97, 0x2d, 0xd9, 0xbf, 0xd0,
      0x86, 0x14, 0x88, 0x06, 0xdf, 0xbc, 0x16, 0x54,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x74, 0xcc, 0xd4, 0x02, 0x73, 0x3b, 0x50, 0xb5,
      0x18, 0x35, 0x72, 0x36, 0xbe, 0xe2, 0x1b, 0x58,
      0x89, 0xa4, 0x84, 0xbe, 0x4d, 0x5b, 0x20, 0xd7,
      0xe9, 0xff, 0xac, 0x92, 0x8b, 0x7a, 0x61, 0x6d,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x97, 0x5f, 0xcb, 0x48, 0x08, 0xa3, 0xdb, 0x00,
      0x94, 0xd8, 0x33, 0xb6, 0x79, 0xa7, 0xd9, 0x1b,
      0x9e, 0x87, 0x70, 0xfb, 0x65, 0x90, 0xff, 0x03,
      0x98, 0x3f, 0x6e, 0x65, 0x81, 0xcf, 0x2e, 0x17,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "shopee_interfaces/msg/detail/pose2_d__functions.h"
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
static const rosidl_type_hash_t shopee_interfaces__msg__Pose2D__EXPECTED_HASH = {1, {
    0x71, 0x2f, 0x30, 0x9e, 0xb8, 0xf2, 0x39, 0x8a,
    0x73, 0x09, 0x3e, 0x1c, 0x51, 0x17, 0x2a, 0x0b,
    0xe0, 0xaf, 0x80, 0x0a, 0x54, 0x8c, 0xd7, 0x9b,
    0xe7, 0xf4, 0xbe, 0x49, 0x0e, 0xeb, 0x1d, 0x90,
  }};
#endif

static char shopee_interfaces__srv__PickeeMobileMoveToLocation__TYPE_NAME[] = "shopee_interfaces/srv/PickeeMobileMoveToLocation";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";
static char shopee_interfaces__msg__Pose2D__TYPE_NAME[] = "shopee_interfaces/msg/Pose2D";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__TYPE_NAME[] = "shopee_interfaces/srv/PickeeMobileMoveToLocation_Event";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__TYPE_NAME[] = "shopee_interfaces/srv/PickeeMobileMoveToLocation_Request";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__TYPE_NAME[] = "shopee_interfaces/srv/PickeeMobileMoveToLocation_Response";

// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeMobileMoveToLocation__FIELD_NAME__request_message[] = "request_message";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation__FIELD_NAME__response_message[] = "response_message";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeMobileMoveToLocation__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__TYPE_NAME, 56, 56},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__TYPE_NAME, 57, 57},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__TYPE_NAME, 54, 54},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PickeeMobileMoveToLocation__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__TYPE_NAME, 54, 54},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__TYPE_NAME, 56, 56},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__TYPE_NAME, 57, 57},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PickeeMobileMoveToLocation__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeMobileMoveToLocation__TYPE_NAME, 48, 48},
      {shopee_interfaces__srv__PickeeMobileMoveToLocation__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PickeeMobileMoveToLocation__REFERENCED_TYPE_DESCRIPTIONS, 6, 6},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__Pose2D__EXPECTED_HASH, shopee_interfaces__msg__Pose2D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__msg__Pose2D__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[5].fields = shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__location_id[] = "location_id";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__target_pose[] = "target_pose";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__global_path[] = "global_path";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__navigation_mode[] = "navigation_mode";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__location_id, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__target_pose, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__global_path, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_UNBOUNDED_SEQUENCE,
      0,
      0,
      {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELD_NAME__navigation_mode, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__TYPE_NAME, 56, 56},
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__FIELDS, 6, 6},
    },
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&shopee_interfaces__msg__Pose2D__EXPECTED_HASH, shopee_interfaces__msg__Pose2D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = shopee_interfaces__msg__Pose2D__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__FIELD_NAME__success[] = "success";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__TYPE_NAME, 57, 57},
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__FIELD_NAME__info[] = "info";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__FIELD_NAME__request[] = "request";
static char shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__TYPE_NAME, 56, 56},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__TYPE_NAME, 57, 57},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Pose2D__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__TYPE_NAME, 56, 56},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__TYPE_NAME, 57, 57},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__TYPE_NAME, 54, 54},
      {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__Pose2D__EXPECTED_HASH, shopee_interfaces__msg__Pose2D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__msg__Pose2D__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "int32 order_id\n"
  "int32 location_id\n"
  "shopee_interfaces/Pose2D target_pose\n"
  "shopee_interfaces/Pose2D[] global_path\n"
  "string navigation_mode\n"
  "---\n"
  "bool success\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeMobileMoveToLocation__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation__TYPE_NAME, 48, 48},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 179, 179},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__TYPE_NAME, 56, 56},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__TYPE_NAME, 57, 57},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__TYPE_NAME, 54, 54},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeMobileMoveToLocation__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[7];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 7, 7};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeMobileMoveToLocation__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__msg__Pose2D__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__get_individual_type_description_source(NULL);
    sources[5] = *shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_individual_type_description_source(NULL);
    sources[6] = *shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__Pose2D__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeMobileMoveToLocation_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__msg__Pose2D__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__srv__PickeeMobileMoveToLocation_Request__get_individual_type_description_source(NULL);
    sources[5] = *shopee_interfaces__srv__PickeeMobileMoveToLocation_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
