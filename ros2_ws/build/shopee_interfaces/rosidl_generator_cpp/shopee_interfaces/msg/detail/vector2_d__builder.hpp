// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/Vector2D.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/vector2_d.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/vector2_d__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_Vector2D_vy
{
public:
  explicit Init_Vector2D_vy(::shopee_interfaces::msg::Vector2D & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::Vector2D vy(::shopee_interfaces::msg::Vector2D::_vy_type arg)
  {
    msg_.vy = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::Vector2D msg_;
};

class Init_Vector2D_vx
{
public:
  Init_Vector2D_vx()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Vector2D_vy vx(::shopee_interfaces::msg::Vector2D::_vx_type arg)
  {
    msg_.vx = std::move(arg);
    return Init_Vector2D_vy(msg_);
  }

private:
  ::shopee_interfaces::msg::Vector2D msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::Vector2D>()
{
  return shopee_interfaces::msg::builder::Init_Vector2D_vx();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__BUILDER_HPP_
