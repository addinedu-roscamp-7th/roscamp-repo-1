// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:srv/PackeeVisionCheckCartPresence.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/srv/detail/packee_vision_check_cart_presence__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PackeeVisionCheckCartPresence__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x7e, 0xad, 0x72, 0x26, 0x58, 0x42, 0x15, 0x12,
      0xe0, 0xa0, 0x18, 0x73, 0x6a, 0x4d, 0x92, 0xf0,
      0xd1, 0xe4, 0xa6, 0xab, 0x76, 0xe8, 0x5f, 0x1e,
      0xf9, 0xe7, 0x5b, 0x80, 0xf6, 0xf0, 0x8f, 0xc4,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xa0, 0x8f, 0xfd, 0x60, 0x9d, 0xd5, 0x3d, 0x41,
      0x02, 0x70, 0x8e, 0x78, 0xe0, 0x2b, 0xf5, 0x75,
      0xdc, 0x8c, 0x85, 0xfb, 0x5a, 0xce, 0x0a, 0x76,
      0x6f, 0x07, 0x8e, 0x33, 0xf6, 0x92, 0xff, 0xa4,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x7c, 0x29, 0x81, 0x31, 0x6a, 0xb2, 0x3e, 0x7f,
      0x7e, 0xf5, 0x69, 0x9c, 0x66, 0x19, 0xcd, 0x8c,
      0x34, 0xec, 0x60, 0x54, 0xf7, 0xbc, 0x48, 0xbf,
      0x1f, 0xaa, 0x41, 0x8f, 0x60, 0x74, 0x42, 0x4d,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x18, 0x21, 0x97, 0xe1, 0xea, 0xce, 0x17, 0x90,
      0xd2, 0x5d, 0xc3, 0x23, 0x14, 0x58, 0x36, 0x5b,
      0xfa, 0xe7, 0x7c, 0x4a, 0x52, 0x65, 0x37, 0x3c,
      0xba, 0xf8, 0xfa, 0xa4, 0x4a, 0xf0, 0x83, 0x06,
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

static char shopee_interfaces__srv__PackeeVisionCheckCartPresence__TYPE_NAME[] = "shopee_interfaces/srv/PackeeVisionCheckCartPresence";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__TYPE_NAME[] = "shopee_interfaces/srv/PackeeVisionCheckCartPresence_Event";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__TYPE_NAME[] = "shopee_interfaces/srv/PackeeVisionCheckCartPresence_Request";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__TYPE_NAME[] = "shopee_interfaces/srv/PackeeVisionCheckCartPresence_Response";

// Define type names, field names, and default values
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence__FIELD_NAME__request_message[] = "request_message";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence__FIELD_NAME__response_message[] = "response_message";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PackeeVisionCheckCartPresence__FIELDS[] = {
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__TYPE_NAME, 59, 59},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__TYPE_NAME, 60, 60},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__TYPE_NAME, 57, 57},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PackeeVisionCheckCartPresence__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__TYPE_NAME, 57, 57},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__TYPE_NAME, 59, 59},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__TYPE_NAME, 60, 60},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PackeeVisionCheckCartPresence__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence__TYPE_NAME, 51, 51},
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__FIELD_NAME__robot_id[] = "robot_id";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__FIELDS[] = {
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__FIELD_NAME__robot_id, 8, 8},
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
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__TYPE_NAME, 59, 59},
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__FIELD_NAME__cart_present[] = "cart_present";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__FIELD_NAME__confidence[] = "confidence";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__FIELDS[] = {
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__FIELD_NAME__cart_present, 12, 12},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__FIELD_NAME__confidence, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__TYPE_NAME, 60, 60},
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__FIELDS, 3, 3},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__FIELD_NAME__info[] = "info";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__FIELD_NAME__request[] = "request";
static char shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__FIELDS[] = {
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__TYPE_NAME, 59, 59},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__TYPE_NAME, 60, 60},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__TYPE_NAME, 59, 59},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__TYPE_NAME, 60, 60},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__TYPE_NAME, 57, 57},
      {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "---\n"
  "bool cart_present\n"
  "float32 confidence\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PackeeVisionCheckCartPresence__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence__TYPE_NAME, 51, 51},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 71, 71},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__TYPE_NAME, 59, 59},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__TYPE_NAME, 60, 60},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__TYPE_NAME, 57, 57},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PackeeVisionCheckCartPresence__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_individual_type_description_source(NULL);
    sources[5] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence_Request__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__srv__PackeeVisionCheckCartPresence_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
