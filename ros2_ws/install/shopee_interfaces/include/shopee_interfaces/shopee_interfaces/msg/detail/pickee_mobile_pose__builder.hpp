// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeMobilePose.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_pose.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_POSE__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_POSE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_mobile_pose__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeMobilePose_status
{
public:
  explicit Init_PickeeMobilePose_status(::shopee_interfaces::msg::PickeeMobilePose & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeMobilePose status(::shopee_interfaces::msg::PickeeMobilePose::_status_type arg)
  {
    msg_.status = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobilePose msg_;
};

class Init_PickeeMobilePose_battery_level
{
public:
  explicit Init_PickeeMobilePose_battery_level(::shopee_interfaces::msg::PickeeMobilePose & msg)
  : msg_(msg)
  {}
  Init_PickeeMobilePose_status battery_level(::shopee_interfaces::msg::PickeeMobilePose::_battery_level_type arg)
  {
    msg_.battery_level = std::move(arg);
    return Init_PickeeMobilePose_status(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobilePose msg_;
};

class Init_PickeeMobilePose_angular_velocity
{
public:
  explicit Init_PickeeMobilePose_angular_velocity(::shopee_interfaces::msg::PickeeMobilePose & msg)
  : msg_(msg)
  {}
  Init_PickeeMobilePose_battery_level angular_velocity(::shopee_interfaces::msg::PickeeMobilePose::_angular_velocity_type arg)
  {
    msg_.angular_velocity = std::move(arg);
    return Init_PickeeMobilePose_battery_level(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobilePose msg_;
};

class Init_PickeeMobilePose_linear_velocity
{
public:
  explicit Init_PickeeMobilePose_linear_velocity(::shopee_interfaces::msg::PickeeMobilePose & msg)
  : msg_(msg)
  {}
  Init_PickeeMobilePose_angular_velocity linear_velocity(::shopee_interfaces::msg::PickeeMobilePose::_linear_velocity_type arg)
  {
    msg_.linear_velocity = std::move(arg);
    return Init_PickeeMobilePose_angular_velocity(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobilePose msg_;
};

class Init_PickeeMobilePose_current_pose
{
public:
  explicit Init_PickeeMobilePose_current_pose(::shopee_interfaces::msg::PickeeMobilePose & msg)
  : msg_(msg)
  {}
  Init_PickeeMobilePose_linear_velocity current_pose(::shopee_interfaces::msg::PickeeMobilePose::_current_pose_type arg)
  {
    msg_.current_pose = std::move(arg);
    return Init_PickeeMobilePose_linear_velocity(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobilePose msg_;
};

class Init_PickeeMobilePose_order_id
{
public:
  explicit Init_PickeeMobilePose_order_id(::shopee_interfaces::msg::PickeeMobilePose & msg)
  : msg_(msg)
  {}
  Init_PickeeMobilePose_current_pose order_id(::shopee_interfaces::msg::PickeeMobilePose::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeMobilePose_current_pose(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobilePose msg_;
};

class Init_PickeeMobilePose_robot_id
{
public:
  Init_PickeeMobilePose_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMobilePose_order_id robot_id(::shopee_interfaces::msg::PickeeMobilePose::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeMobilePose_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobilePose msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeMobilePose>()
{
  return shopee_interfaces::msg::builder::Init_PickeeMobilePose_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_POSE__BUILDER_HPP_
