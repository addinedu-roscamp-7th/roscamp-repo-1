// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeArmTaskStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_arm_task_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARM_TASK_STATUS__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARM_TASK_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_arm_task_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeArmTaskStatus_message
{
public:
  explicit Init_PickeeArmTaskStatus_message(::shopee_interfaces::msg::PickeeArmTaskStatus & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeArmTaskStatus message(::shopee_interfaces::msg::PickeeArmTaskStatus::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArmTaskStatus msg_;
};

class Init_PickeeArmTaskStatus_progress
{
public:
  explicit Init_PickeeArmTaskStatus_progress(::shopee_interfaces::msg::PickeeArmTaskStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeArmTaskStatus_message progress(::shopee_interfaces::msg::PickeeArmTaskStatus::_progress_type arg)
  {
    msg_.progress = std::move(arg);
    return Init_PickeeArmTaskStatus_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArmTaskStatus msg_;
};

class Init_PickeeArmTaskStatus_current_phase
{
public:
  explicit Init_PickeeArmTaskStatus_current_phase(::shopee_interfaces::msg::PickeeArmTaskStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeArmTaskStatus_progress current_phase(::shopee_interfaces::msg::PickeeArmTaskStatus::_current_phase_type arg)
  {
    msg_.current_phase = std::move(arg);
    return Init_PickeeArmTaskStatus_progress(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArmTaskStatus msg_;
};

class Init_PickeeArmTaskStatus_status
{
public:
  explicit Init_PickeeArmTaskStatus_status(::shopee_interfaces::msg::PickeeArmTaskStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeArmTaskStatus_current_phase status(::shopee_interfaces::msg::PickeeArmTaskStatus::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_PickeeArmTaskStatus_current_phase(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArmTaskStatus msg_;
};

class Init_PickeeArmTaskStatus_product_id
{
public:
  explicit Init_PickeeArmTaskStatus_product_id(::shopee_interfaces::msg::PickeeArmTaskStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeArmTaskStatus_status product_id(::shopee_interfaces::msg::PickeeArmTaskStatus::_product_id_type arg)
  {
    msg_.product_id = std::move(arg);
    return Init_PickeeArmTaskStatus_status(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArmTaskStatus msg_;
};

class Init_PickeeArmTaskStatus_order_id
{
public:
  explicit Init_PickeeArmTaskStatus_order_id(::shopee_interfaces::msg::PickeeArmTaskStatus & msg)
  : msg_(msg)
  {}
  Init_PickeeArmTaskStatus_product_id order_id(::shopee_interfaces::msg::PickeeArmTaskStatus::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeArmTaskStatus_product_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArmTaskStatus msg_;
};

class Init_PickeeArmTaskStatus_robot_id
{
public:
  Init_PickeeArmTaskStatus_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeArmTaskStatus_order_id robot_id(::shopee_interfaces::msg::PickeeArmTaskStatus::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeArmTaskStatus_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArmTaskStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeArmTaskStatus>()
{
  return shopee_interfaces::msg::builder::Init_PickeeArmTaskStatus_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARM_TASK_STATUS__BUILDER_HPP_
