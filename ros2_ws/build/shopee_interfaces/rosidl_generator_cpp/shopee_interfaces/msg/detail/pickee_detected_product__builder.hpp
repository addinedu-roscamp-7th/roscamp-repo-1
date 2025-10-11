// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeDetectedProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_detected_product.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_detected_product__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeDetectedProduct_confidence
{
public:
  explicit Init_PickeeDetectedProduct_confidence(::shopee_interfaces::msg::PickeeDetectedProduct & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeDetectedProduct confidence(::shopee_interfaces::msg::PickeeDetectedProduct::_confidence_type arg)
  {
    msg_.confidence = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeDetectedProduct msg_;
};

class Init_PickeeDetectedProduct_bbox_coords
{
public:
  explicit Init_PickeeDetectedProduct_bbox_coords(::shopee_interfaces::msg::PickeeDetectedProduct & msg)
  : msg_(msg)
  {}
  Init_PickeeDetectedProduct_confidence bbox_coords(::shopee_interfaces::msg::PickeeDetectedProduct::_bbox_coords_type arg)
  {
    msg_.bbox_coords = std::move(arg);
    return Init_PickeeDetectedProduct_confidence(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeDetectedProduct msg_;
};

class Init_PickeeDetectedProduct_bbox_number
{
public:
  explicit Init_PickeeDetectedProduct_bbox_number(::shopee_interfaces::msg::PickeeDetectedProduct & msg)
  : msg_(msg)
  {}
  Init_PickeeDetectedProduct_bbox_coords bbox_number(::shopee_interfaces::msg::PickeeDetectedProduct::_bbox_number_type arg)
  {
    msg_.bbox_number = std::move(arg);
    return Init_PickeeDetectedProduct_bbox_coords(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeDetectedProduct msg_;
};

class Init_PickeeDetectedProduct_product_id
{
public:
  Init_PickeeDetectedProduct_product_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeDetectedProduct_bbox_number product_id(::shopee_interfaces::msg::PickeeDetectedProduct::_product_id_type arg)
  {
    msg_.product_id = std::move(arg);
    return Init_PickeeDetectedProduct_bbox_number(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeDetectedProduct msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeDetectedProduct>()
{
  return shopee_interfaces::msg::builder::Init_PickeeDetectedProduct_product_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__BUILDER_HPP_
