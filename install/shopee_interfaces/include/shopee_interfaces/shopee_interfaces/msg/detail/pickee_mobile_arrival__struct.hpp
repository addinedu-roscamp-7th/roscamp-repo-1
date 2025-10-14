// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PickeeMobileArrival.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_mobile_arrival.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'final_pose'
#include "shopee_interfaces/msg/detail/pose2_d__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PickeeMobileArrival __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PickeeMobileArrival __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PickeeMobileArrival_
{
  using Type = PickeeMobileArrival_<ContainerAllocator>;

  explicit PickeeMobileArrival_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : final_pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->location_id = 0l;
      this->position_error = 0.0f;
      this->travel_time = 0.0f;
      this->message = "";
    }
  }

  explicit PickeeMobileArrival_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : final_pose(_alloc, _init),
    message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->location_id = 0l;
      this->position_error = 0.0f;
      this->travel_time = 0.0f;
      this->message = "";
    }
  }

  // field types and members
  using _robot_id_type =
    int32_t;
  _robot_id_type robot_id;
  using _order_id_type =
    int32_t;
  _order_id_type order_id;
  using _location_id_type =
    int32_t;
  _location_id_type location_id;
  using _final_pose_type =
    shopee_interfaces::msg::Pose2D_<ContainerAllocator>;
  _final_pose_type final_pose;
  using _position_error_type =
    float;
  _position_error_type position_error;
  using _travel_time_type =
    float;
  _travel_time_type travel_time;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;

  // setters for named parameter idiom
  Type & set__robot_id(
    const int32_t & _arg)
  {
    this->robot_id = _arg;
    return *this;
  }
  Type & set__order_id(
    const int32_t & _arg)
  {
    this->order_id = _arg;
    return *this;
  }
  Type & set__location_id(
    const int32_t & _arg)
  {
    this->location_id = _arg;
    return *this;
  }
  Type & set__final_pose(
    const shopee_interfaces::msg::Pose2D_<ContainerAllocator> & _arg)
  {
    this->final_pose = _arg;
    return *this;
  }
  Type & set__position_error(
    const float & _arg)
  {
    this->position_error = _arg;
    return *this;
  }
  Type & set__travel_time(
    const float & _arg)
  {
    this->travel_time = _arg;
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
    shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PickeeMobileArrival
    std::shared_ptr<shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PickeeMobileArrival
    std::shared_ptr<shopee_interfaces::msg::PickeeMobileArrival_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeMobileArrival_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->order_id != other.order_id) {
      return false;
    }
    if (this->location_id != other.location_id) {
      return false;
    }
    if (this->final_pose != other.final_pose) {
      return false;
    }
    if (this->position_error != other.position_error) {
      return false;
    }
    if (this->travel_time != other.travel_time) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeMobileArrival_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeMobileArrival_

// alias to use template instance with default allocator
using PickeeMobileArrival =
  shopee_interfaces::msg::PickeeMobileArrival_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_MOBILE_ARRIVAL__STRUCT_HPP_
