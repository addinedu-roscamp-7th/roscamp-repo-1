// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/BBox.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/b_box.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/b_box__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_BBox_y2
{
public:
  explicit Init_BBox_y2(::shopee_interfaces::msg::BBox & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::BBox y2(::shopee_interfaces::msg::BBox::_y2_type arg)
  {
    msg_.y2 = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::BBox msg_;
};

class Init_BBox_x2
{
public:
  explicit Init_BBox_x2(::shopee_interfaces::msg::BBox & msg)
  : msg_(msg)
  {}
  Init_BBox_y2 x2(::shopee_interfaces::msg::BBox::_x2_type arg)
  {
    msg_.x2 = std::move(arg);
    return Init_BBox_y2(msg_);
  }

private:
  ::shopee_interfaces::msg::BBox msg_;
};

class Init_BBox_y1
{
public:
  explicit Init_BBox_y1(::shopee_interfaces::msg::BBox & msg)
  : msg_(msg)
  {}
  Init_BBox_x2 y1(::shopee_interfaces::msg::BBox::_y1_type arg)
  {
    msg_.y1 = std::move(arg);
    return Init_BBox_x2(msg_);
  }

private:
  ::shopee_interfaces::msg::BBox msg_;
};

class Init_BBox_x1
{
public:
  Init_BBox_x1()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_BBox_y1 x1(::shopee_interfaces::msg::BBox::_x1_type arg)
  {
    msg_.x1 = std::move(arg);
    return Init_BBox_y1(msg_);
  }

private:
  ::shopee_interfaces::msg::BBox msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::BBox>()
{
  return shopee_interfaces::msg::builder::Init_BBox_x1();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__B_BOX__BUILDER_HPP_
