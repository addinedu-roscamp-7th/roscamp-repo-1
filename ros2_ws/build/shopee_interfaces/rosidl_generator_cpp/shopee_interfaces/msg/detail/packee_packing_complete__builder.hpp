// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PackeePackingComplete.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_packing_complete.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_PACKING_COMPLETE__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_PACKING_COMPLETE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/packee_packing_complete__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PackeePackingComplete_message
{
public:
  explicit Init_PackeePackingComplete_message(::shopee_interfaces::msg::PackeePackingComplete & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PackeePackingComplete message(::shopee_interfaces::msg::PackeePackingComplete::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeePackingComplete msg_;
};

class Init_PackeePackingComplete_packed_items
{
public:
  explicit Init_PackeePackingComplete_packed_items(::shopee_interfaces::msg::PackeePackingComplete & msg)
  : msg_(msg)
  {}
  Init_PackeePackingComplete_message packed_items(::shopee_interfaces::msg::PackeePackingComplete::_packed_items_type arg)
  {
    msg_.packed_items = std::move(arg);
    return Init_PackeePackingComplete_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeePackingComplete msg_;
};

class Init_PackeePackingComplete_success
{
public:
  explicit Init_PackeePackingComplete_success(::shopee_interfaces::msg::PackeePackingComplete & msg)
  : msg_(msg)
  {}
  Init_PackeePackingComplete_packed_items success(::shopee_interfaces::msg::PackeePackingComplete::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PackeePackingComplete_packed_items(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeePackingComplete msg_;
};

class Init_PackeePackingComplete_order_id
{
public:
  explicit Init_PackeePackingComplete_order_id(::shopee_interfaces::msg::PackeePackingComplete & msg)
  : msg_(msg)
  {}
  Init_PackeePackingComplete_success order_id(::shopee_interfaces::msg::PackeePackingComplete::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PackeePackingComplete_success(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeePackingComplete msg_;
};

class Init_PackeePackingComplete_robot_id
{
public:
  Init_PackeePackingComplete_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeePackingComplete_order_id robot_id(::shopee_interfaces::msg::PackeePackingComplete::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PackeePackingComplete_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeePackingComplete msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PackeePackingComplete>()
{
  return shopee_interfaces::msg::builder::Init_PackeePackingComplete_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_PACKING_COMPLETE__BUILDER_HPP_
