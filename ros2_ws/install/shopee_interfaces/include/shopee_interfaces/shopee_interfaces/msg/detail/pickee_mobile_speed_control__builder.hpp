// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeMobileSpeedControl.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_speed_control.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_mobile_speed_control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeMobileSpeedControl_reason
{
public:
  explicit Init_PickeeMobileSpeedControl_reason(::shopee_interfaces::msg::PickeeMobileSpeedControl & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeMobileSpeedControl reason(::shopee_interfaces::msg::PickeeMobileSpeedControl::_reason_type arg)
  {
    msg_.reason = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileSpeedControl msg_;
};

class Init_PickeeMobileSpeedControl_obstacles
{
public:
  explicit Init_PickeeMobileSpeedControl_obstacles(::shopee_interfaces::msg::PickeeMobileSpeedControl & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileSpeedControl_reason obstacles(::shopee_interfaces::msg::PickeeMobileSpeedControl::_obstacles_type arg)
  {
    msg_.obstacles = std::move(arg);
    return Init_PickeeMobileSpeedControl_reason(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileSpeedControl msg_;
};

class Init_PickeeMobileSpeedControl_target_speed
{
public:
  explicit Init_PickeeMobileSpeedControl_target_speed(::shopee_interfaces::msg::PickeeMobileSpeedControl & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileSpeedControl_obstacles target_speed(::shopee_interfaces::msg::PickeeMobileSpeedControl::_target_speed_type arg)
  {
    msg_.target_speed = std::move(arg);
    return Init_PickeeMobileSpeedControl_obstacles(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileSpeedControl msg_;
};

class Init_PickeeMobileSpeedControl_speed_mode
{
public:
  explicit Init_PickeeMobileSpeedControl_speed_mode(::shopee_interfaces::msg::PickeeMobileSpeedControl & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileSpeedControl_target_speed speed_mode(::shopee_interfaces::msg::PickeeMobileSpeedControl::_speed_mode_type arg)
  {
    msg_.speed_mode = std::move(arg);
    return Init_PickeeMobileSpeedControl_target_speed(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileSpeedControl msg_;
};

class Init_PickeeMobileSpeedControl_order_id
{
public:
  explicit Init_PickeeMobileSpeedControl_order_id(::shopee_interfaces::msg::PickeeMobileSpeedControl & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileSpeedControl_speed_mode order_id(::shopee_interfaces::msg::PickeeMobileSpeedControl::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeMobileSpeedControl_speed_mode(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileSpeedControl msg_;
};

class Init_PickeeMobileSpeedControl_robot_id
{
public:
  Init_PickeeMobileSpeedControl_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMobileSpeedControl_order_id robot_id(::shopee_interfaces::msg::PickeeMobileSpeedControl::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeMobileSpeedControl_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileSpeedControl msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeMobileSpeedControl>()
{
  return shopee_interfaces::msg::builder::Init_PickeeMobileSpeedControl_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_SPEED_CONTROL__BUILDER_HPP_
