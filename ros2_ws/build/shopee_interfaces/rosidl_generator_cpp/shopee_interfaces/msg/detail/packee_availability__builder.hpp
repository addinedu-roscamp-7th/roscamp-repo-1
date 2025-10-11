// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/PackeeAvailability.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_availability.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_AVAILABILITY__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_AVAILABILITY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/packee_availability__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_PackeeAvailability_message
{
public:
  explicit Init_PackeeAvailability_message(::shopee_interfaces::msg::PackeeAvailability & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::PackeeAvailability message(::shopee_interfaces::msg::PackeeAvailability::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeAvailability msg_;
};

class Init_PackeeAvailability_cart_detected
{
public:
  explicit Init_PackeeAvailability_cart_detected(::shopee_interfaces::msg::PackeeAvailability & msg)
  : msg_(msg)
  {}
  Init_PackeeAvailability_message cart_detected(::shopee_interfaces::msg::PackeeAvailability::_cart_detected_type arg)
  {
    msg_.cart_detected = std::move(arg);
    return Init_PackeeAvailability_message(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeAvailability msg_;
};

class Init_PackeeAvailability_available
{
public:
  explicit Init_PackeeAvailability_available(::shopee_interfaces::msg::PackeeAvailability & msg)
  : msg_(msg)
  {}
  Init_PackeeAvailability_cart_detected available(::shopee_interfaces::msg::PackeeAvailability::_available_type arg)
  {
    msg_.available = std::move(arg);
    return Init_PackeeAvailability_cart_detected(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeAvailability msg_;
};

class Init_PackeeAvailability_order_id
{
public:
  explicit Init_PackeeAvailability_order_id(::shopee_interfaces::msg::PackeeAvailability & msg)
  : msg_(msg)
  {}
  Init_PackeeAvailability_available order_id(::shopee_interfaces::msg::PackeeAvailability::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PackeeAvailability_available(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeAvailability msg_;
};

class Init_PackeeAvailability_robot_id
{
public:
  Init_PackeeAvailability_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeAvailability_order_id robot_id(::shopee_interfaces::msg::PackeeAvailability::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PackeeAvailability_order_id(msg_);
  }

private:
  ::shopee_interfaces::msg::PackeeAvailability msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::PackeeAvailability>()
{
  return shopee_interfaces::msg::builder::Init_PackeeAvailability_robot_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_AVAILABILITY__BUILDER_HPP_
