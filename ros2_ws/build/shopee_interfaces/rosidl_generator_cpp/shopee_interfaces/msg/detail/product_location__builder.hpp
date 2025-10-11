// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:msg/ProductLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/product_location.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__BUILDER_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/msg/detail/product_location__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace msg
{

namespace builder
{

class Init_ProductLocation_quantity
{
public:
  explicit Init_ProductLocation_quantity(::shopee_interfaces::msg::ProductLocation & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::msg::ProductLocation quantity(::shopee_interfaces::msg::ProductLocation::_quantity_type arg)
  {
    msg_.quantity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::msg::ProductLocation msg_;
};

class Init_ProductLocation_section_id
{
public:
  explicit Init_ProductLocation_section_id(::shopee_interfaces::msg::ProductLocation & msg)
  : msg_(msg)
  {}
  Init_ProductLocation_quantity section_id(::shopee_interfaces::msg::ProductLocation::_section_id_type arg)
  {
    msg_.section_id = std::move(arg);
    return Init_ProductLocation_quantity(msg_);
  }

private:
  ::shopee_interfaces::msg::ProductLocation msg_;
};

class Init_ProductLocation_location_id
{
public:
  explicit Init_ProductLocation_location_id(::shopee_interfaces::msg::ProductLocation & msg)
  : msg_(msg)
  {}
  Init_ProductLocation_section_id location_id(::shopee_interfaces::msg::ProductLocation::_location_id_type arg)
  {
    msg_.location_id = std::move(arg);
    return Init_ProductLocation_section_id(msg_);
  }

private:
  ::shopee_interfaces::msg::ProductLocation msg_;
};

class Init_ProductLocation_product_id
{
public:
  Init_ProductLocation_product_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ProductLocation_location_id product_id(::shopee_interfaces::msg::ProductLocation::_product_id_type arg)
  {
    msg_.product_id = std::move(arg);
    return Init_ProductLocation_location_id(msg_);
  }

private:
  ::shopee_interfaces::msg::ProductLocation msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::msg::ProductLocation>()
{
  return shopee_interfaces::msg::builder::Init_ProductLocation_product_id();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__BUILDER_HPP_
