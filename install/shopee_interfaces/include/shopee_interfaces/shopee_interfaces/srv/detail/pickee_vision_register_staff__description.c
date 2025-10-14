// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:srv/PickeeVisionRegisterStaff.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/srv/detail/pickee_vision_register_staff__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeVisionRegisterStaff__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x69, 0x42, 0x89, 0xfb, 0x73, 0x1c, 0x9e, 0xb2,
      0x7f, 0xc6, 0x84, 0x5c, 0xa7, 0x4a, 0xdd, 0x0b,
      0xf1, 0xf7, 0xcb, 0xe0, 0xb3, 0x0b, 0xee, 0x57,
      0x05, 0xc7, 0x5f, 0xed, 0xd4, 0xaf, 0x25, 0xee,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x5b, 0xab, 0x09, 0x83, 0x74, 0x5e, 0x2a, 0x78,
      0x34, 0xde, 0xa0, 0xb3, 0x77, 0x47, 0x9e, 0x7c,
      0x92, 0x2f, 0xe7, 0x99, 0x7c, 0x9a, 0xf1, 0x3a,
      0x76, 0x56, 0xc0, 0x96, 0x1c, 0x42, 0xef, 0xdb,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x93, 0xcd, 0x81, 0x11, 0x8a, 0x84, 0xbe, 0x0b,
      0x85, 0x68, 0x30, 0xb6, 0x8e, 0x08, 0x2d, 0x28,
      0xdb, 0x7f, 0xde, 0xef, 0x69, 0xff, 0xcd, 0x93,
      0xa3, 0x5a, 0xad, 0xaf, 0xfa, 0x6a, 0x78, 0xe0,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x30, 0x7a, 0xda, 0x78, 0x51, 0x90, 0xc2, 0xc1,
      0xae, 0xc4, 0xd4, 0x46, 0x54, 0x9e, 0x6b, 0x47,
      0x32, 0x4d, 0x13, 0x79, 0x7e, 0xe4, 0xc8, 0x92,
      0xd7, 0x8e, 0x64, 0x9c, 0xa9, 0x2f, 0x55, 0xa4,
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

static char shopee_interfaces__srv__PickeeVisionRegisterStaff__TYPE_NAME[] = "shopee_interfaces/srv/PickeeVisionRegisterStaff";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__TYPE_NAME[] = "shopee_interfaces/srv/PickeeVisionRegisterStaff_Event";
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__TYPE_NAME[] = "shopee_interfaces/srv/PickeeVisionRegisterStaff_Request";
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__TYPE_NAME[] = "shopee_interfaces/srv/PickeeVisionRegisterStaff_Response";

// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeVisionRegisterStaff__FIELD_NAME__request_message[] = "request_message";
static char shopee_interfaces__srv__PickeeVisionRegisterStaff__FIELD_NAME__response_message[] = "response_message";
static char shopee_interfaces__srv__PickeeVisionRegisterStaff__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeVisionRegisterStaff__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__TYPE_NAME, 55, 55},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__TYPE_NAME, 56, 56},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__TYPE_NAME, 53, 53},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PickeeVisionRegisterStaff__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__TYPE_NAME, 53, 53},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__TYPE_NAME, 55, 55},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__TYPE_NAME, 56, 56},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PickeeVisionRegisterStaff__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeVisionRegisterStaff__TYPE_NAME, 47, 47},
      {shopee_interfaces__srv__PickeeVisionRegisterStaff__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PickeeVisionRegisterStaff__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__FIELD_NAME__robot_id[] = "robot_id";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__FIELD_NAME__robot_id, 8, 8},
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
shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__TYPE_NAME, 55, 55},
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__FIELD_NAME__accepted[] = "accepted";
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__FIELD_NAME__accepted, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__TYPE_NAME, 56, 56},
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__FIELD_NAME__info[] = "info";
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__FIELD_NAME__request[] = "request";
static char shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__FIELDS[] = {
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__TYPE_NAME, 55, 55},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__TYPE_NAME, 56, 56},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__TYPE_NAME, 55, 55},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__TYPE_NAME, 56, 56},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__TYPE_NAME, 53, 53},
      {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "---\n"
  "bool accepted\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeVisionRegisterStaff__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff__TYPE_NAME, 47, 47},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 48, 48},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__TYPE_NAME, 55, 55},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__TYPE_NAME, 56, 56},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__TYPE_NAME, 53, 53},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeVisionRegisterStaff__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeVisionRegisterStaff__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_individual_type_description_source(NULL);
    sources[5] = *shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PickeeVisionRegisterStaff_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__srv__PickeeVisionRegisterStaff_Request__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__srv__PickeeVisionRegisterStaff_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
