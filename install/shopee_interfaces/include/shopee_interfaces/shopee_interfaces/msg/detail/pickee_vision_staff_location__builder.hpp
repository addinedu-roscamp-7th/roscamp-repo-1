// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeVisionStaffLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_staff_location.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_vision_staff_location__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeVisionStaffLocation_is_tracking
{
public:
  explicit Init_PickeeVisionStaffLocation_is_tracking(::shopee_interfaces::msg::PickeeVisionStaffLocation & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeVisionStaffLocation is_tracking(::shopee_interfaces::msg::PickeeVisionStaffLocation::_is_tracking_type arg)
  {
    msg_.is_tracking = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionStaffLocation msg_;
};

class Init_PickeeVisionStaffLocation_distance
{
public:
  explicit Init_PickeeVisionStaffLocation_distance(::shopee_interfaces::msg::PickeeVisionStaffLocation & msg)
  : msg_(msg)
  {}
  Init_PickeeVisionStaffLocation_is_tracking distance(::shopee_interfaces::msg::PickeeVisionStaffLocation::_distance_type arg)
  {
    msg_.distance = std::move(arg);
    return Init_PickeeVisionStaffLocation_is_tracking(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionStaffLocation msg_;
};

class Init_PickeeVisionStaffLocation_relative_position
{
public:
  explicit Init_PickeeVisionStaffLocation_relative_position(::shopee_interfaces::msg::PickeeVisionStaffLocation & msg)
  : msg_(msg)
  {}
  Init_PickeeVisionStaffLocation_distance relative_position(::shopee_interfaces::msg::PickeeVisionStaffLocation::_relative_position_type arg)
  {
    msg_.relative_position = std::move(arg);
    return Init_PickeeVisionStaffLocation_distance(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionStaffLocation msg_;
};

class Init_PickeeVisionStaffLocation_robot_id
{
public:
  Init_PickeeVisionStaffLocation_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeVisionStaffLocation_relative_position robot_id(::shopee_interfaces::msg::PickeeVisionStaffLocation::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeVisionStaffLocation_relative_position(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionStaffLocation msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeVisionStaffLocation>()
{
  return shopee_interfaces::msg::builder::Init_PickeeVisionStaffLocation_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__BUILDER_HPP_
