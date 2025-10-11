// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PackeeRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_robot_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/packee_robot_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PackeeRobotStatus_items_in_cart
{
public:
  explicit Init_PackeeRobotStatus_items_in_cart(::shopee_interfaces::msg::PackeeRobotStatus & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PackeeRobotStatus items_in_cart(::shopee_interfaces::msg::PackeeRobotStatus::_items_in_cart_type arg)
  {
    msg_.items_in_cart = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeRobotStatus msg_;
};

class Init_PackeeRobotStatus_current_order_id
{
public:
  explicit Init_PackeeRobotStatus_current_order_id(::shopee_interfaces::msg::PackeeRobotStatus & msg)
  : msg_(msg)
  {}
  Init_PackeeRobotStatus_items_in_cart current_order_id(::shopee_interfaces::msg::PackeeRobotStatus::_current_order_id_type arg)
  {
    msg_.current_order_id = std::move(arg);
    return Init_PackeeRobotStatus_items_in_cart(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeRobotStatus msg_;
};

class Init_PackeeRobotStatus_state
{
public:
  explicit Init_PackeeRobotStatus_state(::shopee_interfaces::msg::PackeeRobotStatus & msg)
  : msg_(msg)
  {}
  Init_PackeeRobotStatus_current_order_id state(::shopee_interfaces::msg::PackeeRobotStatus::_state_type arg)
  {
    msg_.state = std::move(arg);
    return Init_PackeeRobotStatus_current_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeRobotStatus msg_;
};

class Init_PackeeRobotStatus_robot_id
{
public:
  Init_PackeeRobotStatus_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeRobotStatus_state robot_id(::shopee_interfaces::msg::PackeeRobotStatus::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PackeeRobotStatus_state(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeRobotStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PackeeRobotStatus>()
{
  return shopee_interfaces::msg::builder::Init_PackeeRobotStatus_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__BUILDER_HPP_
