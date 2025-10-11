// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeVisionDetection.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_detection.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_DETECTION__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_DETECTION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_vision_detection__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeVisionDetection_message
{
public:
  explicit Init_PickeeVisionDetection_message(::shopee_interfaces::msg::PickeeVisionDetection & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeVisionDetection message(::shopee_interfaces::msg::PickeeVisionDetection::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionDetection msg_;
};

class Init_PickeeVisionDetection_products
{
public:
  explicit Init_PickeeVisionDetection_products(::shopee_interfaces::msg::PickeeVisionDetection & msg)
  : msg_(msg)
  {}
  Init_PickeeVisionDetection_message products(::shopee_interfaces::msg::PickeeVisionDetection::_products_type arg)
  {
    msg_.products = std::move(arg);
    return Init_PickeeVisionDetection_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionDetection msg_;
};

class Init_PickeeVisionDetection_success
{
public:
  explicit Init_PickeeVisionDetection_success(::shopee_interfaces::msg::PickeeVisionDetection & msg)
  : msg_(msg)
  {}
  Init_PickeeVisionDetection_products success(::shopee_interfaces::msg::PickeeVisionDetection::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PickeeVisionDetection_products(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionDetection msg_;
};

class Init_PickeeVisionDetection_order_id
{
public:
  explicit Init_PickeeVisionDetection_order_id(::shopee_interfaces::msg::PickeeVisionDetection & msg)
  : msg_(msg)
  {}
  Init_PickeeVisionDetection_success order_id(::shopee_interfaces::msg::PickeeVisionDetection::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeVisionDetection_success(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionDetection msg_;
};

class Init_PickeeVisionDetection_robot_id
{
public:
  Init_PickeeVisionDetection_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeVisionDetection_order_id robot_id(::shopee_interfaces::msg::PickeeVisionDetection::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeVisionDetection_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionDetection msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeVisionDetection>()
{
  return shopee_interfaces::msg::builder::Init_PickeeVisionDetection_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_DETECTION__BUILDER_HPP_
