// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/ArmPoseStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/arm_pose_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__ARM_POSE_STATUS__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__ARM_POSE_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/arm_pose_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_ArmPoseStatus_message
{
public:
  explicit Init_ArmPoseStatus_message(::shopee_interfaces::msg::ArmPoseStatus & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::ArmPoseStatus message(::shopee_interfaces::msg::ArmPoseStatus::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::ArmPoseStatus msg_;
};

class Init_ArmPoseStatus_progress
{
public:
  explicit Init_ArmPoseStatus_progress(::shopee_interfaces::msg::ArmPoseStatus & msg)
  : msg_(msg)
  {}
  Init_ArmPoseStatus_message progress(::shopee_interfaces::msg::ArmPoseStatus::_progress_type arg)
  {
    msg_.progress = std::move(arg);
    return Init_ArmPoseStatus_message(msg_);
  }

private:
  ::shopee_interfaces::msg::ArmPoseStatus msg_;
};

class Init_ArmPoseStatus_status
{
public:
  explicit Init_ArmPoseStatus_status(::shopee_interfaces::msg::ArmPoseStatus & msg)
  : msg_(msg)
  {}
  Init_ArmPoseStatus_progress status(::shopee_interfaces::msg::ArmPoseStatus::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_ArmPoseStatus_progress(msg_);
  }

private:
  ::shopee_interfaces::msg::ArmPoseStatus msg_;
};

class Init_ArmPoseStatus_pose_type
{
public:
  explicit Init_ArmPoseStatus_pose_type(::shopee_interfaces::msg::ArmPoseStatus & msg)
  : msg_(msg)
  {}
  Init_ArmPoseStatus_status pose_type(::shopee_interfaces::msg::ArmPoseStatus::_pose_type_type arg)
  {
    msg_.pose_type = std::move(arg);
    return Init_ArmPoseStatus_status(msg_);
  }

private:
  ::shopee_interfaces::msg::ArmPoseStatus msg_;
};

class Init_ArmPoseStatus_order_id
{
public:
  explicit Init_ArmPoseStatus_order_id(::shopee_interfaces::msg::ArmPoseStatus & msg)
  : msg_(msg)
  {}
  Init_ArmPoseStatus_pose_type order_id(::shopee_interfaces::msg::ArmPoseStatus::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_ArmPoseStatus_pose_type(msg_);
  }

private:
  ::shopee_interfaces::msg::ArmPoseStatus msg_;
};

class Init_ArmPoseStatus_robot_id
{
public:
  Init_ArmPoseStatus_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ArmPoseStatus_order_id robot_id(::shopee_interfaces::msg::ArmPoseStatus::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_ArmPoseStatus_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::ArmPoseStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::ArmPoseStatus>()
{
  return shopee_interfaces::msg::builder::Init_ArmPoseStatus_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__ARM_POSE_STATUS__BUILDER_HPP_
