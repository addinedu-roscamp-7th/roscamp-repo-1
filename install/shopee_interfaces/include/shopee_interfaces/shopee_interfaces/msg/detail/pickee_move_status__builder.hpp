// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeMoveStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_move_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOVE_STATUS__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOVE_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_move_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeMoveStatus_location_id
{
public:
  explicit Init_PickeeMoveStatus_location_id(::shopee_interfaces::msg::PickeeMoveStatus & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeMoveStatus location_id(::shopee_interfaces::msg::PickeeMoveStatus::_location_id_type arg)
  {
    msg_.location_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMoveStatus msg_;
};

class Init_PickeeMoveStatus_order_id
{
public:
  explicit Init_PickeeMoveStatus_order_id(::shopee_interfaces::msg::PickeeMoveStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeMoveStatus_location_id order_id(::shopee_interfaces::msg::PickeeMoveStatus::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeMoveStatus_location_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMoveStatus msg_;
};

class Init_PickeeMoveStatus_robot_id
{
public:
  Init_PickeeMoveStatus_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMoveStatus_order_id robot_id(::shopee_interfaces::msg::PickeeMoveStatus::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeMoveStatus_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMoveStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeMoveStatus>()
{
  return shopee_interfaces::msg::builder::Init_PickeeMoveStatus_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOVE_STATUS__BUILDER_HPP_
