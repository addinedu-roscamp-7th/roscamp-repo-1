// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeMobileArrival.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_arrival.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_mobile_arrival__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeMobileArrival_message
{
public:
  explicit Init_PickeeMobileArrival_message(::shopee_interfaces::msg::PickeeMobileArrival & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeMobileArrival message(::shopee_interfaces::msg::PickeeMobileArrival::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileArrival msg_;
};

class Init_PickeeMobileArrival_travel_time
{
public:
  explicit Init_PickeeMobileArrival_travel_time(::shopee_interfaces::msg::PickeeMobileArrival & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileArrival_message travel_time(::shopee_interfaces::msg::PickeeMobileArrival::_travel_time_type arg)
  {
    msg_.travel_time = std::move(arg);
    return Init_PickeeMobileArrival_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileArrival msg_;
};

class Init_PickeeMobileArrival_position_error
{
public:
  explicit Init_PickeeMobileArrival_position_error(::shopee_interfaces::msg::PickeeMobileArrival & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileArrival_travel_time position_error(::shopee_interfaces::msg::PickeeMobileArrival::_position_error_type arg)
  {
    msg_.position_error = std::move(arg);
    return Init_PickeeMobileArrival_travel_time(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileArrival msg_;
};

class Init_PickeeMobileArrival_final_pose
{
public:
  explicit Init_PickeeMobileArrival_final_pose(::shopee_interfaces::msg::PickeeMobileArrival & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileArrival_position_error final_pose(::shopee_interfaces::msg::PickeeMobileArrival::_final_pose_type arg)
  {
    msg_.final_pose = std::move(arg);
    return Init_PickeeMobileArrival_position_error(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileArrival msg_;
};

class Init_PickeeMobileArrival_location_id
{
public:
  explicit Init_PickeeMobileArrival_location_id(::shopee_interfaces::msg::PickeeMobileArrival & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileArrival_final_pose location_id(::shopee_interfaces::msg::PickeeMobileArrival::_location_id_type arg)
  {
    msg_.location_id = std::move(arg);
    return Init_PickeeMobileArrival_final_pose(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileArrival msg_;
};

class Init_PickeeMobileArrival_order_id
{
public:
  explicit Init_PickeeMobileArrival_order_id(::shopee_interfaces::msg::PickeeMobileArrival & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileArrival_location_id order_id(::shopee_interfaces::msg::PickeeMobileArrival::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeMobileArrival_location_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileArrival msg_;
};

class Init_PickeeMobileArrival_robot_id
{
public:
  Init_PickeeMobileArrival_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMobileArrival_order_id robot_id(::shopee_interfaces::msg::PickeeMobileArrival::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeMobileArrival_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeMobileArrival msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeMobileArrival>()
{
  return shopee_interfaces::msg::builder::Init_PickeeMobileArrival_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__BUILDER_HPP_
