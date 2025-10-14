// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from shopee_interfaces:srv/PickeeWorkflowReturnToStaff.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_workflow_return_to_staff.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_WORKFLOW_RETURN_TO_STAFF__BUILDER_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_WORKFLOW_RETURN_TO_STAFF__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "shopee_interfaces/srv/detail/pickee_workflow_return_to_staff__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeWorkflowReturnToStaff_Request_robot_id
{
public:
  Init_PickeeWorkflowReturnToStaff_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Request robot_id(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Request>()
{
  return shopee_interfaces::srv::builder::Init_PickeeWorkflowReturnToStaff_Request_robot_id();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeWorkflowReturnToStaff_Response_message
{
public:
  explicit Init_PickeeWorkflowReturnToStaff_Response_message(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Response & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Response message(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Response msg_;
};

class Init_PickeeWorkflowReturnToStaff_Response_success
{
public:
  Init_PickeeWorkflowReturnToStaff_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeWorkflowReturnToStaff_Response_message success(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PickeeWorkflowReturnToStaff_Response_message(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Response>()
{
  return shopee_interfaces::srv::builder::Init_PickeeWorkflowReturnToStaff_Response_success();
}

}  // namespace shopee_interfaces


namespace shopee_interfaces
{

namespace srv
{

namespace builder
{

class Init_PickeeWorkflowReturnToStaff_Event_response
{
public:
  explicit Init_PickeeWorkflowReturnToStaff_Event_response(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event & msg)
  : msg_(msg)
  {}
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event response(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event msg_;
};

class Init_PickeeWorkflowReturnToStaff_Event_request
{
public:
  explicit Init_PickeeWorkflowReturnToStaff_Event_request(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event & msg)
  : msg_(msg)
  {}
  Init_PickeeWorkflowReturnToStaff_Event_response request(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_PickeeWorkflowReturnToStaff_Event_response(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event msg_;
};

class Init_PickeeWorkflowReturnToStaff_Event_info
{
public:
  Init_PickeeWorkflowReturnToStaff_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PickeeWorkflowReturnToStaff_Event_request info(::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_PickeeWorkflowReturnToStaff_Event_request(msg_);
  }

private:
  ::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::shopee_interfaces::srv::PickeeWorkflowReturnToStaff_Event>()
{
  return shopee_interfaces::srv::builder::Init_PickeeWorkflowReturnToStaff_Event_info();
}

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_WORKFLOW_RETURN_TO_STAFF__BUILDER_HPP_
