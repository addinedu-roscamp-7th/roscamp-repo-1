// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:msg/PickeeVisionStaffRegister.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_staff_register.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__TRAITS_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/msg/detail/pickee_vision_staff_register__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace shopee_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PickeeVisionStaffRegister & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_id
  {
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
    out << ", ";
  }

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
  const PickeeVisionStaffRegister & msg,
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

inline std::string to_yaml(const PickeeVisionStaffRegister & msg, bool use_flow_style = false)
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
  const shopee_interfaces::msg::PickeeVisionStaffRegister & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::msg::PickeeVisionStaffRegister & msg)
{
  return shopee_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::msg::PickeeVisionStaffRegister>()
{
  return "shopee_interfaces::msg::PickeeVisionStaffRegister";
}

template<>
inline const char * name<shopee_interfaces::msg::PickeeVisionStaffRegister>()
{
  return "shopee_interfaces/msg/PickeeVisionStaffRegister";
}

template<>
struct has_fixed_size<shopee_interfaces::msg::PickeeVisionStaffRegister>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<shopee_interfaces::msg::PickeeVisionStaffRegister>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<shopee_interfaces::msg::PickeeVisionStaffRegister>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__TRAITS_HPP_
