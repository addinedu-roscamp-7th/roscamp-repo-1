// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeProductSelection.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_product_selection.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_SELECTION__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_SELECTION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_product_selection__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeProductSelection_message
{
public:
  explicit Init_PickeeProductSelection_message(::shopee_interfaces::msg::PickeeProductSelection & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeProductSelection message(::shopee_interfaces::msg::PickeeProductSelection::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductSelection msg_;
};

class Init_PickeeProductSelection_quantity
{
public:
  explicit Init_PickeeProductSelection_quantity(::shopee_interfaces::msg::PickeeProductSelection & msg)
  : msg_(msg)
  {}
  Init_PickeeProductSelection_message quantity(::shopee_interfaces::msg::PickeeProductSelection::_quantity_type arg)
  {
    msg_.quantity = std::move(arg);
    return Init_PickeeProductSelection_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductSelection msg_;
};

class Init_PickeeProductSelection_success
{
public:
  explicit Init_PickeeProductSelection_success(::shopee_interfaces::msg::PickeeProductSelection & msg)
  : msg_(msg)
  {}
  Init_PickeeProductSelection_quantity success(::shopee_interfaces::msg::PickeeProductSelection::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PickeeProductSelection_quantity(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductSelection msg_;
};

class Init_PickeeProductSelection_product_id
{
public:
  explicit Init_PickeeProductSelection_product_id(::shopee_interfaces::msg::PickeeProductSelection & msg)
  : msg_(msg)
  {}
  Init_PickeeProductSelection_success product_id(::shopee_interfaces::msg::PickeeProductSelection::_product_id_type arg)
  {
    msg_.product_id = std::move(arg);
    return Init_PickeeProductSelection_success(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductSelection msg_;
};

class Init_PickeeProductSelection_order_id
{
public:
  explicit Init_PickeeProductSelection_order_id(::shopee_interfaces::msg::PickeeProductSelection & msg)
  : msg_(msg)
  {}
  Init_PickeeProductSelection_product_id order_id(::shopee_interfaces::msg::PickeeProductSelection::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeProductSelection_product_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductSelection msg_;
};

class Init_PickeeProductSelection_robot_id
{
public:
  Init_PickeeProductSelection_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeProductSelection_order_id robot_id(::shopee_interfaces::msg::PickeeProductSelection::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeProductSelection_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductSelection msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeProductSelection>()
{
  return shopee_interfaces::msg::builder::Init_PickeeProductSelection_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_SELECTION__BUILDER_HPP_
