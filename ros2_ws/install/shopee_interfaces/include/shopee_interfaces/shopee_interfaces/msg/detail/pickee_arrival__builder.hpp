// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeArrival.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_arrival.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARRIVAL__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARRIVAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_arrival__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeArrival_section_id
{
public:
  explicit Init_PickeeArrival_section_id(::shopee_interfaces::msg::PickeeArrival & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeArrival section_id(::shopee_interfaces::msg::PickeeArrival::_section_id_type arg)
  {
    msg_.section_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArrival msg_;
};

class Init_PickeeArrival_location_id
{
public:
  explicit Init_PickeeArrival_location_id(::shopee_interfaces::msg::PickeeArrival & msg)
  : msg_(msg)
  {}
  Init_PickeeArrival_section_id location_id(::shopee_interfaces::msg::PickeeArrival::_location_id_type arg)
  {
    msg_.location_id = std::move(arg);
    return Init_PickeeArrival_section_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArrival msg_;
};

class Init_PickeeArrival_order_id
{
public:
  explicit Init_PickeeArrival_order_id(::shopee_interfaces::msg::PickeeArrival & msg)
  : msg_(msg)
  {}
  Init_PickeeArrival_location_id order_id(::shopee_interfaces::msg::PickeeArrival::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeArrival_location_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArrival msg_;
};

class Init_PickeeArrival_robot_id
{
public:
  Init_PickeeArrival_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeArrival_order_id robot_id(::shopee_interfaces::msg::PickeeArrival::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeArrival_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeArrival msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeArrival>()
{
  return shopee_interfaces::msg::builder::Init_PickeeArrival_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARRIVAL__BUILDER_HPP_
