// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from shopee_interfaces:msg/PackeeDetectedProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_detected_product.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__TRAITS_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "shopee_interfaces/msg/detail/packee_detected_product__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'bbox'
#include "shopee_interfaces/msg/detail/b_box__traits.hpp"
// Member 'position'
#include "shopee_interfaces/msg/detail/point3_d__traits.hpp"

namespace shopee_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PackeeDetectedProduct & msg,
  std::ostream & out)
{
  out << "{";
  // member: product_id
  {
    out << "product_id: ";
    rosidl_generator_traits::value_to_yaml(msg.product_id, out);
    out << ", ";
  }

  // member: bbox
  {
    out << "bbox: ";
    to_flow_style_yaml(msg.bbox, out);
    out << ", ";
  }

  // member: confidence
  {
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << ", ";
  }

  // member: position
  {
    out << "position: ";
    to_flow_style_yaml(msg.position, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PackeeDetectedProduct & msg,
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

  // member: bbox
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bbox:\n";
    to_block_style_yaml(msg.bbox, out, indentation + 2);
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

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position:\n";
    to_block_style_yaml(msg.position, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PackeeDetectedProduct & msg, bool use_flow_style = false)
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
  const shopee_interfaces::msg::PackeeDetectedProduct & msg,
  std::ostream & out, size_t indentation = 0)
{
  shopee_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use shopee_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const shopee_interfaces::msg::PackeeDetectedProduct & msg)
{
  return shopee_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<shopee_interfaces::msg::PackeeDetectedProduct>()
{
  return "shopee_interfaces::msg::PackeeDetectedProduct";
}

template<>
inline const char * name<shopee_interfaces::msg::PackeeDetectedProduct>()
{
  return "shopee_interfaces/msg/PackeeDetectedProduct";
}

template<>
struct has_fixed_size<shopee_interfaces::msg::PackeeDetectedProduct>
  : std::integral_constant<bool, has_fixed_size<shopee_interfaces::msg::BBox>::value && has_fixed_size<shopee_interfaces::msg::Point3D>::value> {};

template<>
struct has_bounded_size<shopee_interfaces::msg::PackeeDetectedProduct>
  : std::integral_constant<bool, has_bounded_size<shopee_interfaces::msg::BBox>::value && has_bounded_size<shopee_interfaces::msg::Point3D>::value> {};

template<>
struct is_message<shopee_interfaces::msg::PackeeDetectedProduct>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__TRAITS_HPP_
