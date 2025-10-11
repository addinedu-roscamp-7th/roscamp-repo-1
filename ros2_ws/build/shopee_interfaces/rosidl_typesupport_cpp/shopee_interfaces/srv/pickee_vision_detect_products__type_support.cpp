// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from shopee_interfaces:srv/PickeeVisionDetectProducts.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "shopee_interfaces/srv/detail/pickee_vision_detect_products__functions.h"
#include "shopee_interfaces/srv/detail/pickee_vision_detect_products__struct.hpp"
#include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
#include "rosidl_typesupport_cpp/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace shopee_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _PickeeVisionDetectProducts_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _PickeeVisionDetectProducts_Request_type_support_ids_t;

static const _PickeeVisionDetectProducts_Request_type_support_ids_t _PickeeVisionDetectProducts_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _PickeeVisionDetectProducts_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _PickeeVisionDetectProducts_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _PickeeVisionDetectProducts_Request_type_support_symbol_names_t _PickeeVisionDetectProducts_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Request)),
  }
};

typedef struct _PickeeVisionDetectProducts_Request_type_support_data_t
{
  void * data[2];
} _PickeeVisionDetectProducts_Request_type_support_data_t;

static _PickeeVisionDetectProducts_Request_type_support_data_t _PickeeVisionDetectProducts_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _PickeeVisionDetectProducts_Request_message_typesupport_map = {
  2,
  "shopee_interfaces",
  &_PickeeVisionDetectProducts_Request_message_typesupport_ids.typesupport_identifier[0],
  &_PickeeVisionDetectProducts_Request_message_typesupport_symbol_names.symbol_name[0],
  &_PickeeVisionDetectProducts_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t PickeeVisionDetectProducts_Request_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_PickeeVisionDetectProducts_Request_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Request__get_type_hash,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Request__get_type_description,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace shopee_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Request>()
{
  return &::shopee_interfaces::srv::rosidl_typesupport_cpp::PickeeVisionDetectProducts_Request_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Request)() {
  return get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Request>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "shopee_interfaces/srv/detail/pickee_vision_detect_products__functions.h"
// already included above
// #include "shopee_interfaces/srv/detail/pickee_vision_detect_products__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace shopee_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _PickeeVisionDetectProducts_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _PickeeVisionDetectProducts_Response_type_support_ids_t;

static const _PickeeVisionDetectProducts_Response_type_support_ids_t _PickeeVisionDetectProducts_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _PickeeVisionDetectProducts_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _PickeeVisionDetectProducts_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _PickeeVisionDetectProducts_Response_type_support_symbol_names_t _PickeeVisionDetectProducts_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Response)),
  }
};

typedef struct _PickeeVisionDetectProducts_Response_type_support_data_t
{
  void * data[2];
} _PickeeVisionDetectProducts_Response_type_support_data_t;

static _PickeeVisionDetectProducts_Response_type_support_data_t _PickeeVisionDetectProducts_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _PickeeVisionDetectProducts_Response_message_typesupport_map = {
  2,
  "shopee_interfaces",
  &_PickeeVisionDetectProducts_Response_message_typesupport_ids.typesupport_identifier[0],
  &_PickeeVisionDetectProducts_Response_message_typesupport_symbol_names.symbol_name[0],
  &_PickeeVisionDetectProducts_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t PickeeVisionDetectProducts_Response_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_PickeeVisionDetectProducts_Response_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Response__get_type_hash,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Response__get_type_description,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace shopee_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Response>()
{
  return &::shopee_interfaces::srv::rosidl_typesupport_cpp::PickeeVisionDetectProducts_Response_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Response)() {
  return get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Response>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "shopee_interfaces/srv/detail/pickee_vision_detect_products__functions.h"
// already included above
// #include "shopee_interfaces/srv/detail/pickee_vision_detect_products__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace shopee_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _PickeeVisionDetectProducts_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _PickeeVisionDetectProducts_Event_type_support_ids_t;

static const _PickeeVisionDetectProducts_Event_type_support_ids_t _PickeeVisionDetectProducts_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _PickeeVisionDetectProducts_Event_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _PickeeVisionDetectProducts_Event_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _PickeeVisionDetectProducts_Event_type_support_symbol_names_t _PickeeVisionDetectProducts_Event_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Event)),
  }
};

typedef struct _PickeeVisionDetectProducts_Event_type_support_data_t
{
  void * data[2];
} _PickeeVisionDetectProducts_Event_type_support_data_t;

static _PickeeVisionDetectProducts_Event_type_support_data_t _PickeeVisionDetectProducts_Event_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _PickeeVisionDetectProducts_Event_message_typesupport_map = {
  2,
  "shopee_interfaces",
  &_PickeeVisionDetectProducts_Event_message_typesupport_ids.typesupport_identifier[0],
  &_PickeeVisionDetectProducts_Event_message_typesupport_symbol_names.symbol_name[0],
  &_PickeeVisionDetectProducts_Event_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t PickeeVisionDetectProducts_Event_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_PickeeVisionDetectProducts_Event_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Event__get_type_hash,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Event__get_type_description,
  &shopee_interfaces__srv__PickeeVisionDetectProducts_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace shopee_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Event>()
{
  return &::shopee_interfaces::srv::rosidl_typesupport_cpp::PickeeVisionDetectProducts_Event_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts_Event)() {
  return get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Event>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "shopee_interfaces/srv/detail/pickee_vision_detect_products__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/service_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace shopee_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _PickeeVisionDetectProducts_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _PickeeVisionDetectProducts_type_support_ids_t;

static const _PickeeVisionDetectProducts_type_support_ids_t _PickeeVisionDetectProducts_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _PickeeVisionDetectProducts_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _PickeeVisionDetectProducts_type_support_symbol_names_t;
#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _PickeeVisionDetectProducts_type_support_symbol_names_t _PickeeVisionDetectProducts_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts)),
  }
};

typedef struct _PickeeVisionDetectProducts_type_support_data_t
{
  void * data[2];
} _PickeeVisionDetectProducts_type_support_data_t;

static _PickeeVisionDetectProducts_type_support_data_t _PickeeVisionDetectProducts_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _PickeeVisionDetectProducts_service_typesupport_map = {
  2,
  "shopee_interfaces",
  &_PickeeVisionDetectProducts_service_typesupport_ids.typesupport_identifier[0],
  &_PickeeVisionDetectProducts_service_typesupport_symbol_names.symbol_name[0],
  &_PickeeVisionDetectProducts_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t PickeeVisionDetectProducts_service_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_PickeeVisionDetectProducts_service_typesupport_map),
  ::rosidl_typesupport_cpp::get_service_typesupport_handle_function,
  ::rosidl_typesupport_cpp::get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Request>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Response>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts_Event>(),
  &::rosidl_typesupport_cpp::service_create_event_message<shopee_interfaces::srv::PickeeVisionDetectProducts>,
  &::rosidl_typesupport_cpp::service_destroy_event_message<shopee_interfaces::srv::PickeeVisionDetectProducts>,
  &shopee_interfaces__srv__PickeeVisionDetectProducts__get_type_hash,
  &shopee_interfaces__srv__PickeeVisionDetectProducts__get_type_description,
  &shopee_interfaces__srv__PickeeVisionDetectProducts__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace shopee_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts>()
{
  return &::shopee_interfaces::srv::rosidl_typesupport_cpp::PickeeVisionDetectProducts_service_type_support_handle;
}

}  // namespace rosidl_typesupport_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_cpp, shopee_interfaces, srv, PickeeVisionDetectProducts)() {
  return ::rosidl_typesupport_cpp::get_service_type_support_handle<shopee_interfaces::srv::PickeeVisionDetectProducts>();
}

#ifdef __cplusplus
}
#endif
