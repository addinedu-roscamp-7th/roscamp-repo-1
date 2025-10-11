// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/Obstacle.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/obstacle.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/obstacle__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_Obstacle_confidence
{
public:
  explicit Init_Obstacle_confidence(::shopee_interfaces::msg::Obstacle & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::Obstacle confidence(::shopee_interfaces::msg::Obstacle::_confidence_type arg)
  {
    msg_.confidence = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::Obstacle msg_;
};

class Init_Obstacle_bbox
{
public:
  explicit Init_Obstacle_bbox(::shopee_interfaces::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_confidence bbox(::shopee_interfaces::msg::Obstacle::_bbox_type arg)
  {
    msg_.bbox = std::move(arg);
    return Init_Obstacle_confidence(msg_);
  }

private:
  ::shopee_interfaces::msg::Obstacle msg_;
};

class Init_Obstacle_direction
{
public:
  explicit Init_Obstacle_direction(::shopee_interfaces::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_bbox direction(::shopee_interfaces::msg::Obstacle::_direction_type arg)
  {
    msg_.direction = std::move(arg);
    return Init_Obstacle_bbox(msg_);
  }

private:
  ::shopee_interfaces::msg::Obstacle msg_;
};

class Init_Obstacle_velocity
{
public:
  explicit Init_Obstacle_velocity(::shopee_interfaces::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_direction velocity(::shopee_interfaces::msg::Obstacle::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_Obstacle_direction(msg_);
  }

private:
  ::shopee_interfaces::msg::Obstacle msg_;
};

class Init_Obstacle_distance
{
public:
  explicit Init_Obstacle_distance(::shopee_interfaces::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_velocity distance(::shopee_interfaces::msg::Obstacle::_distance_type arg)
  {
    msg_.distance = std::move(arg);
    return Init_Obstacle_velocity(msg_);
  }

private:
  ::shopee_interfaces::msg::Obstacle msg_;
};

class Init_Obstacle_position
{
public:
  explicit Init_Obstacle_position(::shopee_interfaces::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_distance position(::shopee_interfaces::msg::Obstacle::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_Obstacle_distance(msg_);
  }

private:
  ::shopee_interfaces::msg::Obstacle msg_;
};

class Init_Obstacle_obstacle_type
{
public:
  Init_Obstacle_obstacle_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Obstacle_position obstacle_type(::shopee_interfaces::msg::Obstacle::_obstacle_type_type arg)
  {
    msg_.obstacle_type = std::move(arg);
    return Init_Obstacle_position(msg_);
  }

private:
  ::shopee_interfaces::msg::Obstacle msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::Obstacle>()
{
  return shopee_interfaces::msg::builder::Init_Obstacle_obstacle_type();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__BUILDER_HPP_
