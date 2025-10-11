// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from shopee_interfaces:srv/PackeeArmPickProduct.idl
// generated code does not contain a copyright notice

#include "shopee_interfaces/srv/detail/packee_arm_pick_product__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PackeeArmPickProduct__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x6f, 0xcd, 0xa7, 0xc4, 0xab, 0xdc, 0x27, 0xba,
      0xf9, 0xe9, 0xd1, 0xdc, 0xfe, 0xc1, 0x86, 0x58,
      0xd6, 0x07, 0xf2, 0x05, 0xb6, 0xdf, 0x22, 0xd0,
      0x58, 0xb3, 0x8c, 0x12, 0x84, 0xf5, 0xbe, 0xdb,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PackeeArmPickProduct_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x6b, 0x2a, 0xa8, 0xe6, 0x73, 0x40, 0xab, 0x04,
      0x7d, 0xaf, 0x31, 0x06, 0xd0, 0xfe, 0x73, 0xe8,
      0x54, 0x5c, 0x62, 0xdc, 0xe4, 0xc8, 0x26, 0x71,
      0xa8, 0x53, 0xbe, 0x14, 0x28, 0x21, 0xaa, 0x27,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PackeeArmPickProduct_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x58, 0x21, 0x28, 0x06, 0x2d, 0x69, 0x4d, 0xc8,
      0x8a, 0x93, 0x41, 0x25, 0x63, 0x5f, 0xf3, 0x04,
      0x55, 0x6a, 0xcd, 0x27, 0xd9, 0xb2, 0x3f, 0x46,
      0x2a, 0xb9, 0x40, 0x73, 0x1e, 0x3d, 0x64, 0x44,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_shopee_interfaces
const rosidl_type_hash_t *
shopee_interfaces__srv__PackeeArmPickProduct_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x55, 0x8d, 0x64, 0xfa, 0x3b, 0xd0, 0xc4, 0xc5,
      0x44, 0xb4, 0x7e, 0x4d, 0xf0, 0x79, 0x94, 0xf1,
      0x0b, 0x0e, 0xd0, 0x24, 0xcd, 0xfb, 0x64, 0x95,
      0xd2, 0x42, 0x4d, 0xea, 0x1d, 0xd9, 0xb2, 0x61,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "service_msgs/msg/detail/service_event_info__functions.h"
#include "shopee_interfaces/msg/detail/point3_d__functions.h"
#include "shopee_interfaces/msg/detail/b_box__functions.h"

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

static char shopee_interfaces__srv__PackeeArmPickProduct__TYPE_NAME[] = "shopee_interfaces/srv/PackeeArmPickProduct";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";
static char shopee_interfaces__msg__BBox__TYPE_NAME[] = "shopee_interfaces/msg/BBox";
static char shopee_interfaces__msg__Point3D__TYPE_NAME[] = "shopee_interfaces/msg/Point3D";
static char shopee_interfaces__srv__PackeeArmPickProduct_Event__TYPE_NAME[] = "shopee_interfaces/srv/PackeeArmPickProduct_Event";
static char shopee_interfaces__srv__PackeeArmPickProduct_Request__TYPE_NAME[] = "shopee_interfaces/srv/PackeeArmPickProduct_Request";
static char shopee_interfaces__srv__PackeeArmPickProduct_Response__TYPE_NAME[] = "shopee_interfaces/srv/PackeeArmPickProduct_Response";

// Define type names, field names, and default values
static char shopee_interfaces__srv__PackeeArmPickProduct__FIELD_NAME__request_message[] = "request_message";
static char shopee_interfaces__srv__PackeeArmPickProduct__FIELD_NAME__response_message[] = "response_message";
static char shopee_interfaces__srv__PackeeArmPickProduct__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PackeeArmPickProduct__FIELDS[] = {
  {
    {shopee_interfaces__srv__PackeeArmPickProduct__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PackeeArmPickProduct_Request__TYPE_NAME, 50, 50},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PackeeArmPickProduct_Response__TYPE_NAME, 51, 51},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__srv__PackeeArmPickProduct_Event__TYPE_NAME, 48, 48},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PackeeArmPickProduct__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Point3D__TYPE_NAME, 29, 29},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Event__TYPE_NAME, 48, 48},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__TYPE_NAME, 50, 50},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Response__TYPE_NAME, 51, 51},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PackeeArmPickProduct__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PackeeArmPickProduct__TYPE_NAME, 42, 42},
      {shopee_interfaces__srv__PackeeArmPickProduct__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PackeeArmPickProduct__REFERENCED_TYPE_DESCRIPTIONS, 7, 7},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__BBox__EXPECTED_HASH, shopee_interfaces__msg__BBox__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__msg__BBox__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__Point3D__EXPECTED_HASH, shopee_interfaces__msg__Point3D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__msg__Point3D__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = shopee_interfaces__srv__PackeeArmPickProduct_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[5].fields = shopee_interfaces__srv__PackeeArmPickProduct_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[6].fields = shopee_interfaces__srv__PackeeArmPickProduct_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__robot_id[] = "robot_id";
static char shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__order_id[] = "order_id";
static char shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__product_id[] = "product_id";
static char shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__arm_side[] = "arm_side";
static char shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__target_position[] = "target_position";
static char shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__bbox[] = "bbox";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELDS[] = {
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__order_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__product_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__arm_side, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__target_position, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__Point3D__TYPE_NAME, 29, 29},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELD_NAME__bbox, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PackeeArmPickProduct_Request__REFERENCED_TYPE_DESCRIPTIONS[] = {
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
shopee_interfaces__srv__PackeeArmPickProduct_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PackeeArmPickProduct_Request__TYPE_NAME, 50, 50},
      {shopee_interfaces__srv__PackeeArmPickProduct_Request__FIELDS, 6, 6},
    },
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__REFERENCED_TYPE_DESCRIPTIONS, 2, 2},
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
// Define type names, field names, and default values
static char shopee_interfaces__srv__PackeeArmPickProduct_Response__FIELD_NAME__accepted[] = "accepted";
static char shopee_interfaces__srv__PackeeArmPickProduct_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PackeeArmPickProduct_Response__FIELDS[] = {
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Response__FIELD_NAME__accepted, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Response__FIELD_NAME__message, 7, 7},
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
shopee_interfaces__srv__PackeeArmPickProduct_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PackeeArmPickProduct_Response__TYPE_NAME, 51, 51},
      {shopee_interfaces__srv__PackeeArmPickProduct_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char shopee_interfaces__srv__PackeeArmPickProduct_Event__FIELD_NAME__info[] = "info";
static char shopee_interfaces__srv__PackeeArmPickProduct_Event__FIELD_NAME__request[] = "request";
static char shopee_interfaces__srv__PackeeArmPickProduct_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field shopee_interfaces__srv__PackeeArmPickProduct_Event__FIELDS[] = {
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PackeeArmPickProduct_Request__TYPE_NAME, 50, 50},
    },
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {shopee_interfaces__srv__PackeeArmPickProduct_Response__TYPE_NAME, 51, 51},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription shopee_interfaces__srv__PackeeArmPickProduct_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__BBox__TYPE_NAME, 26, 26},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__msg__Point3D__TYPE_NAME, 29, 29},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__TYPE_NAME, 50, 50},
    {NULL, 0, 0},
  },
  {
    {shopee_interfaces__srv__PackeeArmPickProduct_Response__TYPE_NAME, 51, 51},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
shopee_interfaces__srv__PackeeArmPickProduct_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {shopee_interfaces__srv__PackeeArmPickProduct_Event__TYPE_NAME, 48, 48},
      {shopee_interfaces__srv__PackeeArmPickProduct_Event__FIELDS, 3, 3},
    },
    {shopee_interfaces__srv__PackeeArmPickProduct_Event__REFERENCED_TYPE_DESCRIPTIONS, 6, 6},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__BBox__EXPECTED_HASH, shopee_interfaces__msg__BBox__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = shopee_interfaces__msg__BBox__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&shopee_interfaces__msg__Point3D__EXPECTED_HASH, shopee_interfaces__msg__Point3D__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = shopee_interfaces__msg__Point3D__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = shopee_interfaces__srv__PackeeArmPickProduct_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[5].fields = shopee_interfaces__srv__PackeeArmPickProduct_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "int32 order_id\n"
  "int32 product_id\n"
  "string arm_side\n"
  "shopee_interfaces/Point3D target_position\n"
  "shopee_interfaces/BBox bbox\n"
  "---\n"
  "bool accepted\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PackeeArmPickProduct__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PackeeArmPickProduct__TYPE_NAME, 42, 42},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 166, 166},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PackeeArmPickProduct_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PackeeArmPickProduct_Request__TYPE_NAME, 50, 50},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PackeeArmPickProduct_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PackeeArmPickProduct_Response__TYPE_NAME, 51, 51},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
shopee_interfaces__srv__PackeeArmPickProduct_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {shopee_interfaces__srv__PackeeArmPickProduct_Event__TYPE_NAME, 48, 48},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PackeeArmPickProduct__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[8];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 8, 8};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PackeeArmPickProduct__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__msg__BBox__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__msg__Point3D__get_individual_type_description_source(NULL);
    sources[5] = *shopee_interfaces__srv__PackeeArmPickProduct_Event__get_individual_type_description_source(NULL);
    sources[6] = *shopee_interfaces__srv__PackeeArmPickProduct_Request__get_individual_type_description_source(NULL);
    sources[7] = *shopee_interfaces__srv__PackeeArmPickProduct_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PackeeArmPickProduct_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[3];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 3, 3};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PackeeArmPickProduct_Request__get_individual_type_description_source(NULL),
    sources[1] = *shopee_interfaces__msg__BBox__get_individual_type_description_source(NULL);
    sources[2] = *shopee_interfaces__msg__Point3D__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PackeeArmPickProduct_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PackeeArmPickProduct_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
shopee_interfaces__srv__PackeeArmPickProduct_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[7];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 7, 7};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *shopee_interfaces__srv__PackeeArmPickProduct_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *shopee_interfaces__msg__BBox__get_individual_type_description_source(NULL);
    sources[4] = *shopee_interfaces__msg__Point3D__get_individual_type_description_source(NULL);
    sources[5] = *shopee_interfaces__srv__PackeeArmPickProduct_Request__get_individual_type_description_source(NULL);
    sources[6] = *shopee_interfaces__srv__PackeeArmPickProduct_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
