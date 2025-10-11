// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/PackeeVisionCheckCartPresence.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/packee_vision_check_cart_presence.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_CHECK_CART_PRESENCE__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_CHECK_CART_PRESENCE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/packee_vision_check_cart_presence__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionCheckCartPresence_Request_robot_id
{
public:
  Init_PackeeVisionCheckCartPresence_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Request robot_id(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Request>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionCheckCartPresence_Request_robot_id();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionCheckCartPresence_Response_message
{
public:
  explicit Init_PackeeVisionCheckCartPresence_Response_message(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response message(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response msg_;
};

class Init_PackeeVisionCheckCartPresence_Response_confidence
{
public:
  explicit Init_PackeeVisionCheckCartPresence_Response_confidence(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionCheckCartPresence_Response_message confidence(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response::_confidence_type arg)
  {
    msg_.confidence = std::move(arg);
    return Init_PackeeVisionCheckCartPresence_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response msg_;
};

class Init_PackeeVisionCheckCartPresence_Response_cart_present
{
public:
  Init_PackeeVisionCheckCartPresence_Response_cart_present()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeVisionCheckCartPresence_Response_confidence cart_present(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response::_cart_present_type arg)
  {
    msg_.cart_present = std::move(arg);
    return Init_PackeeVisionCheckCartPresence_Response_confidence(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Response>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionCheckCartPresence_Response_cart_present();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PackeeVisionCheckCartPresence_Event_response
{
public:
  explicit Init_PackeeVisionCheckCartPresence_Event_response(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event response(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event msg_;
};

class Init_PackeeVisionCheckCartPresence_Event_request
{
public:
  explicit Init_PackeeVisionCheckCartPresence_Event_request(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event & msg)
  : msg_(msg)
  {}
  Init_PackeeVisionCheckCartPresence_Event_response request(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_PackeeVisionCheckCartPresence_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event msg_;
};

class Init_PackeeVisionCheckCartPresence_Event_info
{
public:
  Init_PackeeVisionCheckCartPresence_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PackeeVisionCheckCartPresence_Event_request info(::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_PackeeVisionCheckCartPresence_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PackeeVisionCheckCartPresence_Event>()
{
  return shopee_interfaces::srv::builder::Init_PackeeVisionCheckCartPresence_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PACKEE_VISION_CHECK_CART_PRESENCE__BUILDER_HPP_
