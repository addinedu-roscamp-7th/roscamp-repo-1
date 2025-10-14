// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeVisionObstacles.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_obstacles.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_OBSTACLES__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_OBSTACLES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_vision_obstacles__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeVisionObstacles_message
{
public:
  explicit Init_PickeeVisionObstacles_message(::shopee_interfaces::msg::PickeeVisionObstacles & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeVisionObstacles message(::shopee_interfaces::msg::PickeeVisionObstacles::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionObstacles msg_;
};

class Init_PickeeVisionObstacles_obstacles
{
public:
  explicit Init_PickeeVisionObstacles_obstacles(::shopee_interfaces::msg::PickeeVisionObstacles & msg)
  : msg_(msg)
  {}
  Init_PickeeVisionObstacles_message obstacles(::shopee_interfaces::msg::PickeeVisionObstacles::_obstacles_type arg)
  {
    msg_.obstacles = std::move(arg);
    return Init_PickeeVisionObstacles_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionObstacles msg_;
};

class Init_PickeeVisionObstacles_order_id
{
public:
  explicit Init_PickeeVisionObstacles_order_id(::shopee_interfaces::msg::PickeeVisionObstacles & msg)
  : msg_(msg)
  {}
  Init_PickeeVisionObstacles_obstacles order_id(::shopee_interfaces::msg::PickeeVisionObstacles::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeVisionObstacles_obstacles(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionObstacles msg_;
};

class Init_PickeeVisionObstacles_robot_id
{
public:
  Init_PickeeVisionObstacles_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeVisionObstacles_order_id robot_id(::shopee_interfaces::msg::PickeeVisionObstacles::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeVisionObstacles_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeVisionObstacles msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeVisionObstacles>()
{
  return shopee_interfaces::msg::builder::Init_PickeeVisionObstacles_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_OBSTACLES__BUILDER_HPP_
