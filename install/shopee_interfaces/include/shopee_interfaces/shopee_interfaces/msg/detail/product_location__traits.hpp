// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:msg/ProductLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/product_location.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__TRAITS_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/msg/detail/product_location__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace shopee_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const ProductLocation & msg,
  std::ostream & out)
{
  out << "{";
  // member: product_id
  {
    out << "product_id: ";
    rosidl_generator_traits::value_to_yaml(msg.product_id, out);
    out << ", ";
  }

  // member: location_id
  {
    out << "location_id: ";
    rosidl_generator_traits::value_to_yaml(msg.location_id, out);
    out << ", ";
  }

  // member: section_id
  {
    out << "section_id: ";
    rosidl_generator_traits::value_to_yaml(msg.section_id, out);
    out << ", ";
  }

  // member: quantity
  {
    out << "quantity: ";
    rosidl_generator_traits::value_to_yaml(msg.quantity, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ProductLocation & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: product_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "product_id: ";
    rosidl_generator_traits::value_to_yaml(msg.product_id, out);
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

  // member: section_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "section_id: ";
    rosidl_generator_traits::value_to_yaml(msg.section_id, out);
    out << "\n";
  }

  // member: quantity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "quantity: ";
    rosidl_generator_traits::value_to_yaml(msg.quantity, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ProductLocation & msg, bool use_flow_style = false)
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
  const shopee_interfaces::msg::ProductLocation & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::msg::ProductLocation & msg)
{
  return shopee_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::msg::ProductLocation>()
{
  return "shopee_interfaces::msg::ProductLocation";
}

template<>
inline const char * name<shopee_interfaces::msg::ProductLocation>()
{
  return "shopee_interfaces/msg/ProductLocation";
}

template<>
struct has_fixed_size<shopee_interfaces::msg::ProductLocation>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<shopee_interfaces::msg::ProductLocation>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<shopee_interfaces::msg::ProductLocation>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__TRAITS_HPP_
