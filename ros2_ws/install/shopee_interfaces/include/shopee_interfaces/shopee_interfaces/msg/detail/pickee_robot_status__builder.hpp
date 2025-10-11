// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_robot_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ROBOT_STATUS__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ROBOT_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_robot_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeRobotStatus_orientation_z
{
public:
  explicit Init_PickeeRobotStatus_orientation_z(::shopee_interfaces::msg::PickeeRobotStatus & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeRobotStatus orientation_z(::shopee_interfaces::msg::PickeeRobotStatus::_orientation_z_type arg)
  {
    msg_.orientation_z = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeRobotStatus msg_;
};

class Init_PickeeRobotStatus_position_y
{
public:
  explicit Init_PickeeRobotStatus_position_y(::shopee_interfaces::msg::PickeeRobotStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeRobotStatus_orientation_z position_y(::shopee_interfaces::msg::PickeeRobotStatus::_position_y_type arg)
  {
    msg_.position_y = std::move(arg);
    return Init_PickeeRobotStatus_orientation_z(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeRobotStatus msg_;
};

class Init_PickeeRobotStatus_position_x
{
public:
  explicit Init_PickeeRobotStatus_position_x(::shopee_interfaces::msg::PickeeRobotStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeRobotStatus_position_y position_x(::shopee_interfaces::msg::PickeeRobotStatus::_position_x_type arg)
  {
    msg_.position_x = std::move(arg);
    return Init_PickeeRobotStatus_position_y(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeRobotStatus msg_;
};

class Init_PickeeRobotStatus_current_order_id
{
public:
  explicit Init_PickeeRobotStatus_current_order_id(::shopee_interfaces::msg::PickeeRobotStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeRobotStatus_position_x current_order_id(::shopee_interfaces::msg::PickeeRobotStatus::_current_order_id_type arg)
  {
    msg_.current_order_id = std::move(arg);
    return Init_PickeeRobotStatus_position_x(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeRobotStatus msg_;
};

class Init_PickeeRobotStatus_battery_level
{
public:
  explicit Init_PickeeRobotStatus_battery_level(::shopee_interfaces::msg::PickeeRobotStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeRobotStatus_current_order_id battery_level(::shopee_interfaces::msg::PickeeRobotStatus::_battery_level_type arg)
  {
    msg_.battery_level = std::move(arg);
    return Init_PickeeRobotStatus_current_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeRobotStatus msg_;
};

class Init_PickeeRobotStatus_state
{
public:
  explicit Init_PickeeRobotStatus_state(::shopee_interfaces::msg::PickeeRobotStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeRobotStatus_battery_level state(::shopee_interfaces::msg::PickeeRobotStatus::_state_type arg)
  {
    msg_.state = std::move(arg);
    return Init_PickeeRobotStatus_battery_level(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeRobotStatus msg_;
};

class Init_PickeeRobotStatus_robot_id
{
public:
  Init_PickeeRobotStatus_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeRobotStatus_state robot_id(::shopee_interfaces::msg::PickeeRobotStatus::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeRobotStatus_state(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeRobotStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeRobotStatus>()
{
  return shopee_interfaces::msg::builder::Init_PickeeRobotStatus_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ROBOT_STATUS__BUILDER_HPP_
