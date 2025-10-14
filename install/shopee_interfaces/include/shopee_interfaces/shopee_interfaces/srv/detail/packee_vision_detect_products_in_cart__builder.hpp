// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/PackeeVisionDetectProductsInCart.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/packee_vision_detect_products_in_cart.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_DETECT_PRODUCTS_IN_CART__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_DETECT_PRODUCTS_IN_CART__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/packee_vision_detect_products_in_cart__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionDetectProductsInCart_Request_expected_product_ids
{
public:
  explicit Init_PackeeVisionDetectProductsInCart_Request_expected_product_ids(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request expected_product_ids(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request::_expected_product_ids_type arg)
  {
    msg_.expected_product_ids = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request msg_;
};

class Init_PackeeVisionDetectProductsInCart_Request_order_id
{
public:
  explicit Init_PackeeVisionDetectProductsInCart_Request_order_id(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionDetectProductsInCart_Request_expected_product_ids order_id(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PackeeVisionDetectProductsInCart_Request_expected_product_ids(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request msg_;
};

class Init_PackeeVisionDetectProductsInCart_Request_robot_id
{
public:
  Init_PackeeVisionDetectProductsInCart_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeVisionDetectProductsInCart_Request_order_id robot_id(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PackeeVisionDetectProductsInCart_Request_order_id(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Request>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionDetectProductsInCart_Request_robot_id();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionDetectProductsInCart_Response_message
{
public:
  explicit Init_PackeeVisionDetectProductsInCart_Response_message(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response message(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response msg_;
};

class Init_PackeeVisionDetectProductsInCart_Response_total_detected
{
public:
  explicit Init_PackeeVisionDetectProductsInCart_Response_total_detected(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionDetectProductsInCart_Response_message total_detected(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response::_total_detected_type arg)
  {
    msg_.total_detected = std::move(arg);
    return Init_PackeeVisionDetectProductsInCart_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response msg_;
};

class Init_PackeeVisionDetectProductsInCart_Response_products
{
public:
  explicit Init_PackeeVisionDetectProductsInCart_Response_products(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionDetectProductsInCart_Response_total_detected products(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response::_products_type arg)
  {
    msg_.products = std::move(arg);
    return Init_PackeeVisionDetectProductsInCart_Response_total_detected(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response msg_;
};

class Init_PackeeVisionDetectProductsInCart_Response_success
{
public:
  Init_PackeeVisionDetectProductsInCart_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeVisionDetectProductsInCart_Response_products success(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PackeeVisionDetectProductsInCart_Response_products(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Response>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionDetectProductsInCart_Response_success();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionDetectProductsInCart_Event_response
{
public:
  explicit Init_PackeeVisionDetectProductsInCart_Event_response(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event response(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event msg_;
};

class Init_PackeeVisionDetectProductsInCart_Event_request
{
public:
  explicit Init_PackeeVisionDetectProductsInCart_Event_request(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionDetectProductsInCart_Event_response request(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_PackeeVisionDetectProductsInCart_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event msg_;
};

class Init_PackeeVisionDetectProductsInCart_Event_info
{
public:
  Init_PackeeVisionDetectProductsInCart_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeVisionDetectProductsInCart_Event_request info(::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_PackeeVisionDetectProductsInCart_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionDetectProductsInCart_Event>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionDetectProductsInCart_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_DETECT_PRODUCTS_IN_CART__BUILDER_HPP_
