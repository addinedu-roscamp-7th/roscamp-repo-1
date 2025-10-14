// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PickeeProductLoaded.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_product_loaded.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_LOADED__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_LOADED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/pickee_product_loaded__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PickeeProductLoaded_message
{
public:
  explicit Init_PickeeProductLoaded_message(::shopee_interfaces::msg::PickeeProductLoaded & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PickeeProductLoaded message(::shopee_interfaces::msg::PickeeProductLoaded::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductLoaded msg_;
};

class Init_PickeeProductLoaded_success
{
public:
  explicit Init_PickeeProductLoaded_success(::shopee_interfaces::msg::PickeeProductLoaded & msg)
  : msg_(msg)
  {}
  Init_PickeeProductLoaded_message success(::shopee_interfaces::msg::PickeeProductLoaded::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PickeeProductLoaded_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductLoaded msg_;
};

class Init_PickeeProductLoaded_quantity
{
public:
  explicit Init_PickeeProductLoaded_quantity(::shopee_interfaces::msg::PickeeProductLoaded & msg)
  : msg_(msg)
  {}
  Init_PickeeProductLoaded_success quantity(::shopee_interfaces::msg::PickeeProductLoaded::_quantity_type arg)
  {
    msg_.quantity = std::move(arg);
    return Init_PickeeProductLoaded_success(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductLoaded msg_;
};

class Init_PickeeProductLoaded_product_id
{
public:
  explicit Init_PickeeProductLoaded_product_id(::shopee_interfaces::msg::PickeeProductLoaded & msg)
  : msg_(msg)
  {}
  Init_PickeeProductLoaded_quantity product_id(::shopee_interfaces::msg::PickeeProductLoaded::_product_id_type arg)
  {
    msg_.product_id = std::move(arg);
    return Init_PickeeProductLoaded_quantity(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductLoaded msg_;
};

class Init_PickeeProductLoaded_robot_id
{
public:
  Init_PickeeProductLoaded_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeProductLoaded_product_id robot_id(::shopee_interfaces::msg::PickeeProductLoaded::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeProductLoaded_product_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PickeeProductLoaded msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PickeeProductLoaded>()
{
  return shopee_interfaces::msg::builder::Init_PickeeProductLoaded_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_LOADED__BUILDER_HPP_
