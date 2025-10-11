// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeVisionStaffRegister.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_staff_register.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_vision_staff_register__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeVisionStaffRegister_message
{
public:
  explicit Init_PickeeVisionStaffRegister_message(::shopee_interfaces::msg::PickeeVisionStaffRegister & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeVisionStaffRegister message(::shopee_interfaces::msg::PickeeVisionStaffRegister::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionStaffRegister msg_;
};

class Init_PickeeVisionStaffRegister_success
{
public:
  explicit Init_PickeeVisionStaffRegister_success(::shopee_interfaces::msg::PickeeVisionStaffRegister & msg)
  : msg_(msg)
  {}
  Init_PickeeVisionStaffRegister_message success(::shopee_interfaces::msg::PickeeVisionStaffRegister::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PickeeVisionStaffRegister_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionStaffRegister msg_;
};

class Init_PickeeVisionStaffRegister_robot_id
{
public:
  Init_PickeeVisionStaffRegister_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeVisionStaffRegister_success robot_id(::shopee_interfaces::msg::PickeeVisionStaffRegister::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeVisionStaffRegister_success(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionStaffRegister msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeVisionStaffRegister>()
{
  return shopee_interfaces::msg::builder::Init_PickeeVisionStaffRegister_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_REGISTER__BUILDER_HPP_
