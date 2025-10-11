// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/PackeeArmPlaceProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/packee_arm_place_product.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PLACE_PRODUCT__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PLACE_PRODUCT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/packee_arm_place_product__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeArmPlaceProduct_Request_box_position
{
public:
  explicit Init_PackeeArmPlaceProduct_Request_box_position(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Request box_position(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request::_box_position_type arg)
  {
    msg_.box_position = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Request msg_;
};

class Init_PackeeArmPlaceProduct_Request_arm_side
{
public:
  explicit Init_PackeeArmPlaceProduct_Request_arm_side(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request & msg)
  : msg_(msg)
  {}
  Init_PackeeArmPlaceProduct_Request_box_position arm_side(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request::_arm_side_type arg)
  {
    msg_.arm_side = std::move(arg);
    return Init_PackeeArmPlaceProduct_Request_box_position(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Request msg_;
};

class Init_PackeeArmPlaceProduct_Request_product_id
{
public:
  explicit Init_PackeeArmPlaceProduct_Request_product_id(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request & msg)
  : msg_(msg)
  {}
  Init_PackeeArmPlaceProduct_Request_arm_side product_id(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request::_product_id_type arg)
  {
    msg_.product_id = std::move(arg);
    return Init_PackeeArmPlaceProduct_Request_arm_side(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Request msg_;
};

class Init_PackeeArmPlaceProduct_Request_order_id
{
public:
  explicit Init_PackeeArmPlaceProduct_Request_order_id(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request & msg)
  : msg_(msg)
  {}
  Init_PackeeArmPlaceProduct_Request_product_id order_id(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PackeeArmPlaceProduct_Request_product_id(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Request msg_;
};

class Init_PackeeArmPlaceProduct_Request_robot_id
{
public:
  Init_PackeeArmPlaceProduct_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeArmPlaceProduct_Request_order_id robot_id(::shopee_interfaces::srv::PackeeArmPlaceProduct_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PackeeArmPlaceProduct_Request_order_id(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeArmPlaceProduct_Request>()
{
  return shopee_interfaces::srv::builder::Init_PackeeArmPlaceProduct_Request_robot_id();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeArmPlaceProduct_Response_message
{
public:
  explicit Init_PackeeArmPlaceProduct_Response_message(::shopee_interfaces::srv::PackeeArmPlaceProduct_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Response message(::shopee_interfaces::srv::PackeeArmPlaceProduct_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Response msg_;
};

class Init_PackeeArmPlaceProduct_Response_accepted
{
public:
  Init_PackeeArmPlaceProduct_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeArmPlaceProduct_Response_message accepted(::shopee_interfaces::srv::PackeeArmPlaceProduct_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_PackeeArmPlaceProduct_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeArmPlaceProduct_Response>()
{
  return shopee_interfaces::srv::builder::Init_PackeeArmPlaceProduct_Response_accepted();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeArmPlaceProduct_Event_response
{
public:
  explicit Init_PackeeArmPlaceProduct_Event_response(::shopee_interfaces::srv::PackeeArmPlaceProduct_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Event response(::shopee_interfaces::srv::PackeeArmPlaceProduct_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Event msg_;
};

class Init_PackeeArmPlaceProduct_Event_request
{
public:
  explicit Init_PackeeArmPlaceProduct_Event_request(::shopee_interfaces::srv::PackeeArmPlaceProduct_Event & msg)
  : msg_(msg)
  {}
  Init_PackeeArmPlaceProduct_Event_response request(::shopee_interfaces::srv::PackeeArmPlaceProduct_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_PackeeArmPlaceProduct_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Event msg_;
};

class Init_PackeeArmPlaceProduct_Event_info
{
public:
  Init_PackeeArmPlaceProduct_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeArmPlaceProduct_Event_request info(::shopee_interfaces::srv::PackeeArmPlaceProduct_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_PackeeArmPlaceProduct_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeArmPlaceProduct_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeArmPlaceProduct_Event>()
{
  return shopee_interfaces::srv::builder::Init_PackeeArmPlaceProduct_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_ARM_PLACE_PRODUCT__BUILDER_HPP_
