// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:srv/PickeeMobileUpdateGlobalPath.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_mobile_update_global_path.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MOBILE_UPDATE_GLOBAL_PATH__TRAITS_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MOBILE_UPDATE_GLOBAL_PATH__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/srv/detail/pickee_mobile_update_global_path__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'global_path'
#include "shopee_interfaces/msg/detail/pose2_d__traits.hpp"

namespace shopee_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const PickeeMobileUpdateGlobalPath_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_id
  {
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
    out << ", ";
  }

  // member: order_id
  {
    out << "order_id: ";
    rosidl_generator_traits::value_to_yaml(msg.order_id, out);
    out << ", ";
  }

  // member: location_id
  {
    out << "location_id: ";
    rosidl_generator_traits::value_to_yaml(msg.location_id, out);
    out << ", ";
  }

  // member: global_path
  {
    if (msg.global_path.size() == 0) {
      out << "global_path: []";
    } else {
      out << "global_path: [";
      size_t pending_items = msg.global_path.size();
      for (auto item : msg.global_path) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PickeeMobileUpdateGlobalPath_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: robot_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
    out << "\n";
  }

  // member: order_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "order_id: ";
    rosidl_generator_traits::value_to_yaml(msg.order_id, out);
    out << "\n";
  }

  // member: location_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "location_id: ";
    rosidl_generator_traits::value_to_yaml(msg.location_id, out);
    out << "\n";
  }

  // member: global_path
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.global_path.size() == 0) {
      out << "global_path: []\n";
    } else {
      out << "global_path:\n";
      for (auto item : msg.global_path) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PickeeMobileUpdateGlobalPath_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace shopee_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use shopee_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request & msg)
{
  return shopee_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>()
{
  return "shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request";
}

template<>
inline const char * name<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>()
{
  return "shopee_interfaces/srv/PickeeMobileUpdateGlobalPath_Request";
}

template<>
struct has_fixed_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace shopee_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const PickeeMobileUpdateGlobalPath_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PickeeMobileUpdateGlobalPath_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PickeeMobileUpdateGlobalPath_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace shopee_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use shopee_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response & msg)
{
  return shopee_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>()
{
  return "shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response";
}

template<>
inline const char * name<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>()
{
  return "shopee_interfaces/srv/PickeeMobileUpdateGlobalPath_Response";
}

template<>
struct has_fixed_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace shopee_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const PickeeMobileUpdateGlobalPath_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PickeeMobileUpdateGlobalPath_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PickeeMobileUpdateGlobalPath_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace shopee_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use shopee_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event & msg)
{
  return shopee_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event>()
{
  return "shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event";
}

template<>
inline const char * name<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event>()
{
  return "shopee_interfaces/srv/PickeeMobileUpdateGlobalPath_Event";
}

template<>
struct has_fixed_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event>
  : std::integral_constant<bool, has_bounded_size<service_msgs::msg::ServiceEventInfo>::value && has_bounded_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>::value && has_bounded_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>::value> {};

template<>
struct is_message<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>()
{
  return "shopee_interfaces::srv::PickeeMobileUpdateGlobalPath";
}

template<>
inline const char * name<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>()
{
  return "shopee_interfaces/srv/PickeeMobileUpdateGlobalPath";
}

template<>
struct has_fixed_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>
  : std::integral_constant<
    bool,
    has_fixed_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>::value &&
    has_fixed_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>::value
  >
{
};

template<>
struct has_bounded_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>
  : std::integral_constant<
    bool,
    has_bounded_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>::value &&
    has_bounded_size<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>::value
  >
{
};

template<>
struct is_service<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath>
  : std::true_type
{
};

template<>
struct is_service_request<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>
  : std::true_type
{
};

template<>
struct is_service_response<shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MOBILE_UPDATE_GLOBAL_PATH__TRAITS_HPP_
