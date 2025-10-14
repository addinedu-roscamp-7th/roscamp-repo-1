// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PackeeDetectedProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_detected_product.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/packee_detected_product__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PackeeDetectedProduct_position
{
public:
  explicit Init_PackeeDetectedProduct_position(::shopee_interfaces::msg::PackeeDetectedProduct & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PackeeDetectedProduct position(::shopee_interfaces::msg::PackeeDetectedProduct::_position_type arg)
  {
    msg_.position = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeDetectedProduct msg_;
};

class Init_PackeeDetectedProduct_confidence
{
public:
  explicit Init_PackeeDetectedProduct_confidence(::shopee_interfaces::msg::PackeeDetectedProduct & msg)
  : msg_(msg)
  {}
  Init_PackeeDetectedProduct_position confidence(::shopee_interfaces::msg::PackeeDetectedProduct::_confidence_type arg)
  {
    msg_.confidence = std::move(arg);
    return Init_PackeeDetectedProduct_position(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeDetectedProduct msg_;
};

class Init_PackeeDetectedProduct_bbox
{
public:
  explicit Init_PackeeDetectedProduct_bbox(::shopee_interfaces::msg::PackeeDetectedProduct & msg)
  : msg_(msg)
  {}
  Init_PackeeDetectedProduct_confidence bbox(::shopee_interfaces::msg::PackeeDetectedProduct::_bbox_type arg)
  {
    msg_.bbox = std::move(arg);
    return Init_PackeeDetectedProduct_confidence(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeDetectedProduct msg_;
};

class Init_PackeeDetectedProduct_product_id
{
public:
  Init_PackeeDetectedProduct_product_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeDetectedProduct_bbox product_id(::shopee_interfaces::msg::PackeeDetectedProduct::_product_id_type arg)
  {
    msg_.product_id = std::move(arg);
    return Init_PackeeDetectedProduct_bbox(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeDetectedProduct msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PackeeDetectedProduct>()
{
  return shopee_interfaces::msg::builder::Init_PackeeDetectedProduct_product_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__BUILDER_HPP_
