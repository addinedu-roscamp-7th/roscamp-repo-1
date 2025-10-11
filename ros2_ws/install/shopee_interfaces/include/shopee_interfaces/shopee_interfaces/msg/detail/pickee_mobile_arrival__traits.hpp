// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:msg/PickeeMobileArrival.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_arrival.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__TRAITS_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/msg/detail/pickee_mobile_arrival__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'final_pose'
#include "shopee_interfaces/msg/detail/pose2_d__traits.hpp"

namespace shopee_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PickeeMobileArrival & msg,
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

  // member: final_pose
  {
    out << "final_pose: ";
    to_flow_style_yaml(msg.final_pose, out);
    out << ", ";
  }

  // member: position_error
  {
    out << "position_error: ";
    rosidl_generator_traits::value_to_yaml(msg.position_error, out);
    out << ", ";
  }

  // member: travel_time
  {
    out << "travel_time: ";
    rosidl_generator_traits::value_to_yaml(msg.travel_time, out);
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
  const PickeeMobileArrival & msg,
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

  // member: final_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "final_pose:\n";
    to_block_style_yaml(msg.final_pose, out, indentation + 2);
  }

  // member: position_error
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position_error: ";
    rosidl_generator_traits::value_to_yaml(msg.position_error, out);
    out << "\n";
  }

  // member: travel_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "travel_time: ";
    rosidl_generator_traits::value_to_yaml(msg.travel_time, out);
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

inline std::string to_yaml(const PickeeMobileArrival & msg, bool use_flow_style = false)
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
  const shopee_interfaces::msg::PickeeMobileArrival & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::msg::PickeeMobileArrival & msg)
{
  return shopee_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::msg::PickeeMobileArrival>()
{
  return "shopee_interfaces::msg::PickeeMobileArrival";
}

template<>
inline const char * name<shopee_interfaces::msg::PickeeMobileArrival>()
{
  return "shopee_interfaces/msg/PickeeMobileArrival";
}

template<>
struct has_fixed_size<shopee_interfaces::msg::PickeeMobileArrival>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<shopee_interfaces::msg::PickeeMobileArrival>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<shopee_interfaces::msg::PickeeMobileArrival>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__TRAITS_HPP_
