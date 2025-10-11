// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/MainGetProductLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/main_get_product_location.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_PRODUCT_LOCATION__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_PRODUCT_LOCATION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/main_get_product_location__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_MainGetProductLocation_Request_product_id
{
public:
  Init_MainGetProductLocation_Request_product_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::shopee_interfaces::srv::MainGetProductLocation_Request product_id(::shopee_interfaces::srv::MainGetProductLocation_Request::_product_id_type arg)
  {
    msg_.product_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetProductLocation_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::MainGetProductLocation_Request>()
{
  return shopee_interfaces::srv::builder::Init_MainGetProductLocation_Request_product_id();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_MainGetProductLocation_Response_message
{
public:
  explicit Init_MainGetProductLocation_Response_message(::shopee_interfaces::srv::MainGetProductLocation_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::MainGetProductLocation_Response message(::shopee_interfaces::srv::MainGetProductLocation_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetProductLocation_Response msg_;
};

class Init_MainGetProductLocation_Response_section_id
{
public:
  explicit Init_MainGetProductLocation_Response_section_id(::shopee_interfaces::srv::MainGetProductLocation_Response & msg)
  : msg_(msg)
  {}
  Init_MainGetProductLocation_Response_message section_id(::shopee_interfaces::srv::MainGetProductLocation_Response::_section_id_type arg)
  {
    msg_.section_id = std::move(arg);
    return Init_MainGetProductLocation_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetProductLocation_Response msg_;
};

class Init_MainGetProductLocation_Response_warehouse_id
{
public:
  explicit Init_MainGetProductLocation_Response_warehouse_id(::shopee_interfaces::srv::MainGetProductLocation_Response & msg)
  : msg_(msg)
  {}
  Init_MainGetProductLocation_Response_section_id warehouse_id(::shopee_interfaces::srv::MainGetProductLocation_Response::_warehouse_id_type arg)
  {
    msg_.warehouse_id = std::move(arg);
    return Init_MainGetProductLocation_Response_section_id(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetProductLocation_Response msg_;
};

class Init_MainGetProductLocation_Response_success
{
public:
  Init_MainGetProductLocation_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MainGetProductLocation_Response_warehouse_id success(::shopee_interfaces::srv::MainGetProductLocation_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_MainGetProductLocation_Response_warehouse_id(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetProductLocation_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::MainGetProductLocation_Response>()
{
  return shopee_interfaces::srv::builder::Init_MainGetProductLocation_Response_success();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_MainGetProductLocation_Event_response
{
public:
  explicit Init_MainGetProductLocation_Event_response(::shopee_interfaces::srv::MainGetProductLocation_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::MainGetProductLocation_Event response(::shopee_interfaces::srv::MainGetProductLocation_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetProductLocation_Event msg_;
};

class Init_MainGetProductLocation_Event_request
{
public:
  explicit Init_MainGetProductLocation_Event_request(::shopee_interfaces::srv::MainGetProductLocation_Event & msg)
  : msg_(msg)
  {}
  Init_MainGetProductLocation_Event_response request(::shopee_interfaces::srv::MainGetProductLocation_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_MainGetProductLocation_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetProductLocation_Event msg_;
};

class Init_MainGetProductLocation_Event_info
{
public:
  Init_MainGetProductLocation_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MainGetProductLocation_Event_request info(::shopee_interfaces::srv::MainGetProductLocation_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_MainGetProductLocation_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetProductLocation_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::MainGetProductLocation_Event>()
{
  return shopee_interfaces::srv::builder::Init_MainGetProductLocation_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_PRODUCT_LOCATION__BUILDER_HPP_
