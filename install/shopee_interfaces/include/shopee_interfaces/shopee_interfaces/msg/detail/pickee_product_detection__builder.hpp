// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeProductDetection.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_product_detection.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_DETECTION__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_DETECTION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_product_detection__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeProductDetection_products
{
public:
  explicit Init_PickeeProductDetection_products(::shopee_interfaces::msg::PickeeProductDetection & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeProductDetection products(::shopee_interfaces::msg::PickeeProductDetection::_products_type arg)
  {
    msg_.products = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductDetection msg_;
};

class Init_PickeeProductDetection_order_id
{
public:
  explicit Init_PickeeProductDetection_order_id(::shopee_interfaces::msg::PickeeProductDetection & msg)
  : msg_(msg)
  {}
  Init_PickeeProductDetection_products order_id(::shopee_interfaces::msg::PickeeProductDetection::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeProductDetection_products(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductDetection msg_;
};

class Init_PickeeProductDetection_robot_id
{
public:
  Init_PickeeProductDetection_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeProductDetection_order_id robot_id(::shopee_interfaces::msg::PickeeProductDetection::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeProductDetection_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductDetection msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeProductDetection>()
{
  return shopee_interfaces::msg::builder::Init_PickeeProductDetection_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_DETECTION__BUILDER_HPP_
