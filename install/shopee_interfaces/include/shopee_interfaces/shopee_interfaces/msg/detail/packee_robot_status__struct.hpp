// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PackeeRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_robot_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PackeeRobotStatus __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PackeeRobotStatus __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PackeeRobotStatus_
{
  using Type = PackeeRobotStatus_<ContainerAllocator>;

  explicit PackeeRobotStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->state = "";
      this->current_order_id = 0l;
      this->items_in_cart = 0l;
    }
  }

  explicit PackeeRobotStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : state(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->state = "";
      this->current_order_id = 0l;
      this->items_in_cart = 0l;
    }
  }

  // field types and members
  using _robot_id_type =
    int32_t;
  _robot_id_type robot_id;
  using _state_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _state_type state;
  using _current_order_id_type =
    int32_t;
  _current_order_id_type current_order_id;
  using _items_in_cart_type =
    int32_t;
  _items_in_cart_type items_in_cart;

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
  Type & set__current_order_id(
    const int32_t & _arg)
  {
    this->current_order_id = _arg;
    return *this;
  }
  Type & set__items_in_cart(
    const int32_t & _arg)
  {
    this->items_in_cart = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PackeeRobotStatus
    std::shared_ptr<shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PackeeRobotStatus
    std::shared_ptr<shopee_interfaces::msg::PackeeRobotStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PackeeRobotStatus_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->state != other.state) {
      return false;
    }
    if (this->current_order_id != other.current_order_id) {
      return false;
    }
    if (this->items_in_cart != other.items_in_cart) {
      return false;
    }
    return true;
  }
  bool operator!=(const PackeeRobotStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PackeeRobotStatus_

// alias to use template instance with default allocator
using PackeeRobotStatus =
  shopee_interfaces::msg::PackeeRobotStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ROBOT_STATUS__STRUCT_HPP_
