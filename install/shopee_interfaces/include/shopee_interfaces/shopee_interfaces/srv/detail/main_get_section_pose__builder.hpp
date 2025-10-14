// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/MainGetSectionPose.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/main_get_section_pose.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_SECTION_POSE__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_SECTION_POSE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/main_get_section_pose__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_MainGetSectionPose_Request_section_id
{
public:
  Init_MainGetSectionPose_Request_section_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::shopee_interfaces::srv::MainGetSectionPose_Request section_id(::shopee_interfaces::srv::MainGetSectionPose_Request::_section_id_type arg)
  {
    msg_.section_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetSectionPose_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::MainGetSectionPose_Request>()
{
  return shopee_interfaces::srv::builder::Init_MainGetSectionPose_Request_section_id();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_MainGetSectionPose_Response_message
{
public:
  explicit Init_MainGetSectionPose_Response_message(::shopee_interfaces::srv::MainGetSectionPose_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::MainGetSectionPose_Response message(::shopee_interfaces::srv::MainGetSectionPose_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetSectionPose_Response msg_;
};

class Init_MainGetSectionPose_Response_success
{
public:
  explicit Init_MainGetSectionPose_Response_success(::shopee_interfaces::srv::MainGetSectionPose_Response & msg)
  : msg_(msg)
  {}
  Init_MainGetSectionPose_Response_message success(::shopee_interfaces::srv::MainGetSectionPose_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_MainGetSectionPose_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetSectionPose_Response msg_;
};

class Init_MainGetSectionPose_Response_pose
{
public:
  Init_MainGetSectionPose_Response_pose()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MainGetSectionPose_Response_success pose(::shopee_interfaces::srv::MainGetSectionPose_Response::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return Init_MainGetSectionPose_Response_success(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetSectionPose_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::MainGetSectionPose_Response>()
{
  return shopee_interfaces::srv::builder::Init_MainGetSectionPose_Response_pose();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_MainGetSectionPose_Event_response
{
public:
  explicit Init_MainGetSectionPose_Event_response(::shopee_interfaces::srv::MainGetSectionPose_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::MainGetSectionPose_Event response(::shopee_interfaces::srv::MainGetSectionPose_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetSectionPose_Event msg_;
};

class Init_MainGetSectionPose_Event_request
{
public:
  explicit Init_MainGetSectionPose_Event_request(::shopee_interfaces::srv::MainGetSectionPose_Event & msg)
  : msg_(msg)
  {}
  Init_MainGetSectionPose_Event_response request(::shopee_interfaces::srv::MainGetSectionPose_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_MainGetSectionPose_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetSectionPose_Event msg_;
};

class Init_MainGetSectionPose_Event_info
{
public:
  Init_MainGetSectionPose_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MainGetSectionPose_Event_request info(::shopee_interfaces::srv::MainGetSectionPose_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_MainGetSectionPose_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::MainGetSectionPose_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::MainGetSectionPose_Event>()
{
  return shopee_interfaces::srv::builder::Init_MainGetSectionPose_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__MAIN_GET_SECTION_POSE__BUILDER_HPP_
