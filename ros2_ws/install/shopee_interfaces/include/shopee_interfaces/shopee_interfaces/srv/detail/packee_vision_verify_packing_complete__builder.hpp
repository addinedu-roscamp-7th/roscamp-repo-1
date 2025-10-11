// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/PackeeVisionVerifyPackingComplete.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/packee_vision_verify_packing_complete.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_VERIFY_PACKING_COMPLETE__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_VERIFY_PACKING_COMPLETE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/packee_vision_verify_packing_complete__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionVerifyPackingComplete_Request_order_id
{
public:
  explicit Init_PackeeVisionVerifyPackingComplete_Request_order_id(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Request & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Request order_id(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Request::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Request msg_;
};

class Init_PackeeVisionVerifyPackingComplete_Request_robot_id
{
public:
  Init_PackeeVisionVerifyPackingComplete_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeVisionVerifyPackingComplete_Request_order_id robot_id(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PackeeVisionVerifyPackingComplete_Request_order_id(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Request>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionVerifyPackingComplete_Request_robot_id();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionVerifyPackingComplete_Response_message
{
public:
  explicit Init_PackeeVisionVerifyPackingComplete_Response_message(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response message(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response msg_;
};

class Init_PackeeVisionVerifyPackingComplete_Response_remaining_product_ids
{
public:
  explicit Init_PackeeVisionVerifyPackingComplete_Response_remaining_product_ids(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionVerifyPackingComplete_Response_message remaining_product_ids(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response::_remaining_product_ids_type arg)
  {
    msg_.remaining_product_ids = std::move(arg);
    return Init_PackeeVisionVerifyPackingComplete_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response msg_;
};

class Init_PackeeVisionVerifyPackingComplete_Response_remaining_items
{
public:
  explicit Init_PackeeVisionVerifyPackingComplete_Response_remaining_items(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionVerifyPackingComplete_Response_remaining_product_ids remaining_items(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response::_remaining_items_type arg)
  {
    msg_.remaining_items = std::move(arg);
    return Init_PackeeVisionVerifyPackingComplete_Response_remaining_product_ids(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response msg_;
};

class Init_PackeeVisionVerifyPackingComplete_Response_cart_empty
{
public:
  Init_PackeeVisionVerifyPackingComplete_Response_cart_empty()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeVisionVerifyPackingComplete_Response_remaining_items cart_empty(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response::_cart_empty_type arg)
  {
    msg_.cart_empty = std::move(arg);
    return Init_PackeeVisionVerifyPackingComplete_Response_remaining_items(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Response>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionVerifyPackingComplete_Response_cart_empty();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionVerifyPackingComplete_Event_response
{
public:
  explicit Init_PackeeVisionVerifyPackingComplete_Event_response(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event response(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event msg_;
};

class Init_PackeeVisionVerifyPackingComplete_Event_request
{
public:
  explicit Init_PackeeVisionVerifyPackingComplete_Event_request(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionVerifyPackingComplete_Event_response request(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_PackeeVisionVerifyPackingComplete_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event msg_;
};

class Init_PackeeVisionVerifyPackingComplete_Event_info
{
public:
  Init_PackeeVisionVerifyPackingComplete_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeVisionVerifyPackingComplete_Event_request info(::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_PackeeVisionVerifyPackingComplete_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionVerifyPackingComplete_Event>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionVerifyPackingComplete_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_VERIFY_PACKING_COMPLETE__BUILDER_HPP_
