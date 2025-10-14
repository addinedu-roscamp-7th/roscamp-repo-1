// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/PickeeMainVideoStreamStart.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_main_video_stream_start.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MAIN_VIDEO_STREAM_START__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MAIN_VIDEO_STREAM_START__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/pickee_main_video_stream_start__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeMainVideoStreamStart_Request_robot_id
{
public:
  explicit Init_PickeeMainVideoStreamStart_Request_robot_id(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request robot_id(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request msg_;
};

class Init_PickeeMainVideoStreamStart_Request_user_id
{
public:
  explicit Init_PickeeMainVideoStreamStart_Request_user_id(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request & msg)
  : msg_(msg)
  {}
  Init_PickeeMainVideoStreamStart_Request_robot_id user_id(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request::_user_id_type arg)
  {
    msg_.user_id = std::move(arg);
    return Init_PickeeMainVideoStreamStart_Request_robot_id(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request msg_;
};

class Init_PickeeMainVideoStreamStart_Request_user_type
{
public:
  Init_PickeeMainVideoStreamStart_Request_user_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMainVideoStreamStart_Request_user_id user_type(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request::_user_type_type arg)
  {
    msg_.user_type = std::move(arg);
    return Init_PickeeMainVideoStreamStart_Request_user_id(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeMainVideoStreamStart_Request>()
{
  return shopee_interfaces::srv::builder::Init_PickeeMainVideoStreamStart_Request_user_type();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeMainVideoStreamStart_Response_message
{
public:
  explicit Init_PickeeMainVideoStreamStart_Response_message(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Response message(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Response msg_;
};

class Init_PickeeMainVideoStreamStart_Response_success
{
public:
  Init_PickeeMainVideoStreamStart_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMainVideoStreamStart_Response_message success(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PickeeMainVideoStreamStart_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeMainVideoStreamStart_Response>()
{
  return shopee_interfaces::srv::builder::Init_PickeeMainVideoStreamStart_Response_success();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeMainVideoStreamStart_Event_response
{
public:
  explicit Init_PickeeMainVideoStreamStart_Event_response(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event response(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event msg_;
};

class Init_PickeeMainVideoStreamStart_Event_request
{
public:
  explicit Init_PickeeMainVideoStreamStart_Event_request(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event & msg)
  : msg_(msg)
  {}
  Init_PickeeMainVideoStreamStart_Event_response request(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_PickeeMainVideoStreamStart_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event msg_;
};

class Init_PickeeMainVideoStreamStart_Event_info
{
public:
  Init_PickeeMainVideoStreamStart_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeMainVideoStreamStart_Event_request info(::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_PickeeMainVideoStreamStart_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeMainVideoStreamStart_Event>()
{
  return shopee_interfaces::srv::builder::Init_PickeeMainVideoStreamStart_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_MAIN_VIDEO_STREAM_START__BUILDER_HPP_
