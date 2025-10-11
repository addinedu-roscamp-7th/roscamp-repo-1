// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:msg/PickeeMobileSpeedControl.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_speed_control.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__TRAITS_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/msg/detail/pickee_mobile_speed_control__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'obstacles'
#include "shopee_interfaces/msg/detail/obstacle__traits.hpp"

namespace shopee_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PickeeMobileSpeedControl & msg,
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

  // member: speed_mode
  {
    out << "speed_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.speed_mode, out);
    out << ", ";
  }

  // member: target_speed
  {
    out << "target_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.target_speed, out);
    out << ", ";
  }

  // member: obstacles
  {
    if (msg.obstacles.size() == 0) {
      out << "obstacles: []";
    } else {
      out << "obstacles: [";
      size_t pending_items = msg.obstacles.size();
      for (auto item : msg.obstacles) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: reason
  {
    out << "reason: ";
    rosidl_generator_traits::value_to_yaml(msg.reason, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PickeeMobileSpeedControl & msg,
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

  // member: speed_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "speed_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.speed_mode, out);
    out << "\n";
  }

  // member: target_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.target_speed, out);
    out << "\n";
  }

  // member: obstacles
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.obstacles.size() == 0) {
      out << "obstacles: []\n";
    } else {
      out << "obstacles:\n";
      for (auto item : msg.obstacles) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: reason
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reason: ";
    rosidl_generator_traits::value_to_yaml(msg.reason, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PickeeMobileSpeedControl & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace shopee_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use shopee_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const shopee_interfaces::msg::PickeeMobileSpeedControl & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::msg::PickeeMobileSpeedControl & msg)
{
  return shopee_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::msg::PickeeMobileSpeedControl>()
{
  return "shopee_interfaces::msg::PickeeMobileSpeedControl";
}

template<>
inline const char * name<shopee_interfaces::msg::PickeeMobileSpeedControl>()
{
  return "shopee_interfaces/msg/PickeeMobileSpeedControl";
}

template<>
struct has_fixed_size<shopee_interfaces::msg::PickeeMobileSpeedControl>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<shopee_interfaces::msg::PickeeMobileSpeedControl>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<shopee_interfaces::msg::PickeeMobileSpeedControl>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__TRAITS_HPP_
