// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PickeeRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_robot_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ROBOT_STATUS__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ROBOT_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PickeeRobotStatus __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PickeeRobotStatus __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PickeeRobotStatus_
{
  using Type = PickeeRobotStatus_<ContainerAllocator>;

  explicit PickeeRobotStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->state = "";
      this->battery_level = 0.0f;
      this->current_order_id = 0l;
      this->position_x = 0.0f;
      this->position_y = 0.0f;
      this->orientation_z = 0.0f;
    }
  }

  explicit PickeeRobotStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : state(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->state = "";
      this->battery_level = 0.0f;
      this->current_order_id = 0l;
      this->position_x = 0.0f;
      this->position_y = 0.0f;
      this->orientation_z = 0.0f;
    }
  }

  // field types and members
  using _robot_id_type =
    int32_t;
  _robot_id_type robot_id;
  using _state_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _state_type state;
  using _battery_level_type =
    float;
  _battery_level_type battery_level;
  using _current_order_id_type =
    int32_t;
  _current_order_id_type current_order_id;
  using _position_x_type =
    float;
  _position_x_type position_x;
  using _position_y_type =
    float;
  _position_y_type position_y;
  using _orientation_z_type =
    float;
  _orientation_z_type orientation_z;

  // setters for named parameter idiom
  Type & set__robot_id(
    const int32_t & _arg)
  {
    this->robot_id = _arg;
    return *this;
  }
  Type & set__state(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->state = _arg;
    return *this;
  }
  Type & set__battery_level(
    const float & _arg)
  {
    this->battery_level = _arg;
    return *this;
  }
  Type & set__current_order_id(
    const int32_t & _arg)
  {
    this->current_order_id = _arg;
    return *this;
  }
  Type & set__position_x(
    const float & _arg)
  {
    this->position_x = _arg;
    return *this;
  }
  Type & set__position_y(
    const float & _arg)
  {
    this->position_y = _arg;
    return *this;
  }
  Type & set__orientation_z(
    const float & _arg)
  {
    this->orientation_z = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PickeeRobotStatus
    std::shared_ptr<shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PickeeRobotStatus
    std::shared_ptr<shopee_interfaces::msg::PickeeRobotStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeRobotStatus_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->state != other.state) {
      return false;
    }
    if (this->battery_level != other.battery_level) {
      return false;
    }
    if (this->current_order_id != other.current_order_id) {
      return false;
    }
    if (this->position_x != other.position_x) {
      return false;
    }
    if (this->position_y != other.position_y) {
      return false;
    }
    if (this->orientation_z != other.orientation_z) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeRobotStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeRobotStatus_

// alias to use template instance with default allocator
using PickeeRobotStatus =
  shopee_interfaces::msg::PickeeRobotStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ROBOT_STATUS__STRUCT_HPP_
