// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeCartHandover.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_cart_handover.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_CART_HANDOVER__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_CART_HANDOVER__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_cart_handover__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeCartHandover_order_id
{
public:
  explicit Init_PickeeCartHandover_order_id(::shopee_interfaces::msg::PickeeCartHandover & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeCartHandover order_id(::shopee_interfaces::msg::PickeeCartHandover::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeCartHandover msg_;
};

class Init_PickeeCartHandover_robot_id
{
public:
  Init_PickeeCartHandover_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeCartHandover_order_id robot_id(::shopee_interfaces::msg::PickeeCartHandover::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeCartHandover_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeCartHandover msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeCartHandover>()
{
  return shopee_interfaces::msg::builder::Init_PickeeCartHandover_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_CART_HANDOVER__BUILDER_HPP_
