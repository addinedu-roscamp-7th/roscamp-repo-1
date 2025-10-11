// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:msg/PackeeRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_robot_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__TRAITS_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/msg/detail/packee_robot_status__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace shopee_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PackeeRobotStatus & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_id
  {
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
    out << ", ";
  }

  // member: state
  {
    out << "state: ";
    rosidl_generator_traits::value_to_yaml(msg.state, out);
    out << ", ";
  }

  // member: current_order_id
  {
    out << "current_order_id: ";
    rosidl_generator_traits::value_to_yaml(msg.current_order_id, out);
    out << ", ";
  }

  // member: items_in_cart
  {
    out << "items_in_cart: ";
    rosidl_generator_traits::value_to_yaml(msg.items_in_cart, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PackeeRobotStatus & msg,
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

  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "state: ";
    rosidl_generator_traits::value_to_yaml(msg.state, out);
    out << "\n";
  }

  // member: current_order_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_order_id: ";
    rosidl_generator_traits::value_to_yaml(msg.current_order_id, out);
    out << "\n";
  }

  // member: items_in_cart
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "items_in_cart: ";
    rosidl_generator_traits::value_to_yaml(msg.items_in_cart, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PackeeRobotStatus & msg, bool use_flow_style = false)
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
  const shopee_interfaces::msg::PackeeRobotStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::msg::PackeeRobotStatus & msg)
{
  return shopee_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::msg::PackeeRobotStatus>()
{
  return "shopee_interfaces::msg::PackeeRobotStatus";
}

template<>
inline const char * name<shopee_interfaces::msg::PackeeRobotStatus>()
{
  return "shopee_interfaces/msg/PackeeRobotStatus";
}

template<>
struct has_fixed_size<shopee_interfaces::msg::PackeeRobotStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<shopee_interfaces::msg::PackeeRobotStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<shopee_interfaces::msg::PackeeRobotStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__TRAITS_HPP_
