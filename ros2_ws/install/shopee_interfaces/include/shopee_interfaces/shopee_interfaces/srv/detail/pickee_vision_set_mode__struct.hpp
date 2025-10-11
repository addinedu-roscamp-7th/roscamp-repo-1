// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:srv/PickeeVisionSetMode.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_vision_set_mode.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_VISION_SET_MODE__STRUCT_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_VISION_SET_MODE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Request __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Request __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PickeeVisionSetMode_Request_
{
  using Type = PickeeVisionSetMode_Request_<ContainerAllocator>;

  explicit PickeeVisionSetMode_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->mode = "";
    }
  }

  explicit PickeeVisionSetMode_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mode(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->mode = "";
    }
  }

  // field types and members
  using _robot_id_type =
    int32_t;
  _robot_id_type robot_id;
  using _mode_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _mode_type mode;

  // setters for named parameter idiom
  Type & set__robot_id(
    const int32_t & _arg)
  {
    this->robot_id = _arg;
    return *this;
  }
  Type & set__mode(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Request
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Request
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeVisionSetMode_Request_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeVisionSetMode_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeVisionSetMode_Request_

// alias to use template instance with default allocator
using PickeeVisionSetMode_Request =
  shopee_interfaces::srv::PickeeVisionSetMode_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace shopee_interfaces


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Response __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Response __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PickeeVisionSetMode_Response_
{
  using Type = PickeeVisionSetMode_Response_<ContainerAllocator>;

  explicit PickeeVisionSetMode_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
    }
  }

  explicit PickeeVisionSetMode_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->message = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Response
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Response
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeVisionSetMode_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeVisionSetMode_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeVisionSetMode_Response_

// alias to use template instance with default allocator
using PickeeVisionSetMode_Response =
  shopee_interfaces::srv::PickeeVisionSetMode_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace shopee_interfaces


// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Event __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Event __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PickeeVisionSetMode_Event_
{
  using Type = PickeeVisionSetMode_Event_<ContainerAllocator>;

  explicit PickeeVisionSetMode_Event_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : info(_init)
  {
    (void)_init;
  }

  explicit PickeeVisionSetMode_Event_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : info(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _info_type =
    service_msgs::msg::ServiceEventInfo_<ContainerAllocator>;
  _info_type info;
  using _request_type =
    rosidl_runtime_cpp::BoundedVector<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>>>;
  _request_type request;
  using _response_type =
    rosidl_runtime_cpp::BoundedVector<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>>>;
  _response_type response;

  // setters for named parameter idiom
  Type & set__info(
    const service_msgs::msg::ServiceEventInfo_<ContainerAllocator> & _arg)
  {
    this->info = _arg;
    return *this;
  }
  Type & set__request(
    const rosidl_runtime_cpp::BoundedVector<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::srv::PickeeVisionSetMode_Request_<ContainerAllocator>>> & _arg)
  {
    this->request = _arg;
    return *this;
  }
  Type & set__response(
    const rosidl_runtime_cpp::BoundedVector<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::srv::PickeeVisionSetMode_Response_<ContainerAllocator>>> & _arg)
  {
    this->response = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Event
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__srv__PickeeVisionSetMode_Event
    std::shared_ptr<shopee_interfaces::srv::PickeeVisionSetMode_Event_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeVisionSetMode_Event_ & other) const
  {
    if (this->info != other.info) {
      return false;
    }
    if (this->request != other.request) {
      return false;
    }
    if (this->response != other.response) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeVisionSetMode_Event_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeVisionSetMode_Event_

// alias to use template instance with default allocator
using PickeeVisionSetMode_Event =
  shopee_interfaces::srv::PickeeVisionSetMode_Event_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace shopee_interfaces

namespace shopee_interfaces
{

namespace srv
{

struct PickeeVisionSetMode
{
  using Request = shopee_interfaces::srv::PickeeVisionSetMode_Request;
  using Response = shopee_interfaces::srv::PickeeVisionSetMode_Response;
  using Event = shopee_interfaces::srv::PickeeVisionSetMode_Event;
};

}  // namespace srv

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_VISION_SET_MODE__STRUCT_HPP_
