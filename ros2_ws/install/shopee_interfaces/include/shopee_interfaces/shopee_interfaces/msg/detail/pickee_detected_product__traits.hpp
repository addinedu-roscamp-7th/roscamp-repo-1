// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:msg/PickeeDetectedProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_detected_product.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__TRAITS_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/msg/detail/pickee_detected_product__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'bbox_coords'
#include "shopee_interfaces/msg/detail/b_box__traits.hpp"

namespace shopee_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PickeeDetectedProduct & msg,
  std::ostream & out)
{
  out << "{";
  // member: product_id
  {
    out << "product_id: ";
    rosidl_generator_traits::value_to_yaml(msg.product_id, out);
    out << ", ";
  }

  // member: bbox_number
  {
    out << "bbox_number: ";
    rosidl_generator_traits::value_to_yaml(msg.bbox_number, out);
    out << ", ";
  }

  // member: bbox_coords
  {
    out << "bbox_coords: ";
    to_flow_style_yaml(msg.bbox_coords, out);
    out << ", ";
  }

  // member: confidence
  {
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PickeeDetectedProduct & msg,
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

  // member: bbox_number
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bbox_number: ";
    rosidl_generator_traits::value_to_yaml(msg.bbox_number, out);
    out << "\n";
  }

  // member: bbox_coords
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bbox_coords:\n";
    to_block_style_yaml(msg.bbox_coords, out, indentation + 2);
  }

  // member: confidence
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PickeeDetectedProduct & msg, bool use_flow_style = false)
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
  const shopee_interfaces::msg::PickeeDetectedProduct & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::msg::PickeeDetectedProduct & msg)
{
  return shopee_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::msg::PickeeDetectedProduct>()
{
  return "shopee_interfaces::msg::PickeeDetectedProduct";
}

template<>
inline const char * name<shopee_interfaces::msg::PickeeDetectedProduct>()
{
  return "shopee_interfaces/msg/PickeeDetectedProduct";
}

template<>
struct has_fixed_size<shopee_interfaces::msg::PickeeDetectedProduct>
  : std::integral_constant<bool, has_fixed_size<shopee_interfaces::msg::BBox>::value> {};

template<>
struct has_bounded_size<shopee_interfaces::msg::PickeeDetectedProduct>
  : std::integral_constant<bool, has_bounded_size<shopee_interfaces::msg::BBox>::value> {};

template<>
struct is_message<shopee_interfaces::msg::PickeeDetectedProduct>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__TRAITS_HPP_
