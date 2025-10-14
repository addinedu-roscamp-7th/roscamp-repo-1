// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:msg/PickeeVisionStaffLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_staff_location.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__TRAITS_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/msg/detail/pickee_vision_staff_location__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'relative_position'
#include "shopee_interfaces/msg/detail/point2_d__traits.hpp"

namespace shopee_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PickeeVisionStaffLocation & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_id
  {
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
    out << ", ";
  }

  // member: relative_position
  {
    out << "relative_position: ";
    to_flow_style_yaml(msg.relative_position, out);
    out << ", ";
  }

  // member: distance
  {
    out << "distance: ";
    rosidl_generator_traits::value_to_yaml(msg.distance, out);
    out << ", ";
  }

  // member: is_tracking
  {
    out << "is_tracking: ";
    rosidl_generator_traits::value_to_yaml(msg.is_tracking, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PickeeVisionStaffLocation & msg,
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

  // member: relative_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "relative_position:\n";
    to_block_style_yaml(msg.relative_position, out, indentation + 2);
  }

  // member: distance
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "distance: ";
    rosidl_generator_traits::value_to_yaml(msg.distance, out);
    out << "\n";
  }

  // member: is_tracking
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_tracking: ";
    rosidl_generator_traits::value_to_yaml(msg.is_tracking, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PickeeVisionStaffLocation & msg, bool use_flow_style = false)
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
  const shopee_interfaces::msg::PickeeVisionStaffLocation & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::msg::PickeeVisionStaffLocation & msg)
{
  return shopee_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::msg::PickeeVisionStaffLocation>()
{
  return "shopee_interfaces::msg::PickeeVisionStaffLocation";
}

template<>
inline const char * name<shopee_interfaces::msg::PickeeVisionStaffLocation>()
{
  return "shopee_interfaces/msg/PickeeVisionStaffLocation";
}

template<>
struct has_fixed_size<shopee_interfaces::msg::PickeeVisionStaffLocation>
  : std::integral_constant<bool, has_fixed_size<shopee_interfaces::msg::Point2D>::value> {};

template<>
struct has_bounded_size<shopee_interfaces::msg::PickeeVisionStaffLocation>
  : std::integral_constant<bool, has_bounded_size<shopee_interfaces::msg::Point2D>::value> {};

template<>
struct is_message<shopee_interfaces::msg::PickeeVisionStaffLocation>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__TRAITS_HPP_
