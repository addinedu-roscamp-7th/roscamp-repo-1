// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/PickeeMobileUpdateGlobalPath.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_mobile_update_global_path.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MOBILE_UPDATE_GLOBAL_PATH__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MOBILE_UPDATE_GLOBAL_PATH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/pickee_mobile_update_global_path__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeMobileUpdateGlobalPath_Request_global_path
{
public:
  explicit Init_PickeeMobileUpdateGlobalPath_Request_global_path(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request global_path(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request::_global_path_type arg)
  {
    msg_.global_path = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request msg_;
};

class Init_PickeeMobileUpdateGlobalPath_Request_location_id
{
public:
  explicit Init_PickeeMobileUpdateGlobalPath_Request_location_id(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileUpdateGlobalPath_Request_global_path location_id(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request::_location_id_type arg)
  {
    msg_.location_id = std::move(arg);
    return Init_PickeeMobileUpdateGlobalPath_Request_global_path(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request msg_;
};

class Init_PickeeMobileUpdateGlobalPath_Request_order_id
{
public:
  explicit Init_PickeeMobileUpdateGlobalPath_Request_order_id(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileUpdateGlobalPath_Request_location_id order_id(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request::_order_id_type arg)
  {
    msg_.order_id = std::move(arg);
    return Init_PickeeMobileUpdateGlobalPath_Request_location_id(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request msg_;
};

class Init_PickeeMobileUpdateGlobalPath_Request_robot_id
{
public:
  Init_PickeeMobileUpdateGlobalPath_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMobileUpdateGlobalPath_Request_order_id robot_id(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_PickeeMobileUpdateGlobalPath_Request_order_id(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Request>()
{
  return shopee_interfaces::srv::builder::Init_PickeeMobileUpdateGlobalPath_Request_robot_id();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeMobileUpdateGlobalPath_Response_message
{
public:
  explicit Init_PickeeMobileUpdateGlobalPath_Response_message(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response message(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response msg_;
};

class Init_PickeeMobileUpdateGlobalPath_Response_success
{
public:
  Init_PickeeMobileUpdateGlobalPath_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMobileUpdateGlobalPath_Response_message success(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PickeeMobileUpdateGlobalPath_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Response>()
{
  return shopee_interfaces::srv::builder::Init_PickeeMobileUpdateGlobalPath_Response_success();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeMobileUpdateGlobalPath_Event_response
{
public:
  explicit Init_PickeeMobileUpdateGlobalPath_Event_response(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event response(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event msg_;
};

class Init_PickeeMobileUpdateGlobalPath_Event_request
{
public:
  explicit Init_PickeeMobileUpdateGlobalPath_Event_request(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event & msg)
  : msg_(msg)
  {}
  Init_PickeeMobileUpdateGlobalPath_Event_response request(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_PickeeMobileUpdateGlobalPath_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event msg_;
};

class Init_PickeeMobileUpdateGlobalPath_Event_info
{
public:
  Init_PickeeMobileUpdateGlobalPath_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMobileUpdateGlobalPath_Event_request info(::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_PickeeMobileUpdateGlobalPath_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeMobileUpdateGlobalPath_Event>()
{
  return shopee_interfaces::srv::builder::Init_PickeeMobileUpdateGlobalPath_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MOBILE_UPDATE_GLOBAL_PATH__BUILDER_HPP_
