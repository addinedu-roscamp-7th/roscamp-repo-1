// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PackeeArmTaskStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_arm_task_status.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ARM_TASK_STATUS__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ARM_TASK_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PackeeArmTaskStatus __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PackeeArmTaskStatus __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PackeeArmTaskStatus_
{
  using Type = PackeeArmTaskStatus_<ContainerAllocator>;

  explicit PackeeArmTaskStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->product_id = 0l;
      this->arm_side = "";
      this->status = "";
      this->current_phase = "";
      this->progress = 0.0f;
      this->message = "";
    }
  }

  explicit PackeeArmTaskStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : arm_side(_alloc),
    status(_alloc),
    current_phase(_alloc),
    message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->product_id = 0l;
      this->arm_side = "";
      this->status = "";
      this->current_phase = "";
      this->progress = 0.0f;
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
  using _product_id_type =
    int32_t;
  _product_id_type product_id;
  using _arm_side_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _arm_side_type arm_side;
  using _status_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _status_type status;
  using _current_phase_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _current_phase_type current_phase;
  using _progress_type =
    float;
  _progress_type progress;
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
  Type & set__product_id(
    const int32_t & _arg)
  {
    this->product_id = _arg;
    return *this;
  }
  Type & set__arm_side(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->arm_side = _arg;
    return *this;
  }
  Type & set__status(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__current_phase(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->current_phase = _arg;
    return *this;
  }
  Type & set__progress(
    const float & _arg)
  {
    this->progress = _arg;
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
    shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PackeeArmTaskStatus
    std::shared_ptr<shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PackeeArmTaskStatus
    std::shared_ptr<shopee_interfaces::msg::PackeeArmTaskStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PackeeArmTaskStatus_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->order_id != other.order_id) {
      return false;
    }
    if (this->product_id != other.product_id) {
      return false;
    }
    if (this->arm_side != other.arm_side) {
      return false;
    }
    if (this->status != other.status) {
      return false;
    }
    if (this->current_phase != other.current_phase) {
      return false;
    }
    if (this->progress != other.progress) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const PackeeArmTaskStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PackeeArmTaskStatus_

// alias to use template instance with default allocator
using PackeeArmTaskStatus =
  shopee_interfaces::msg::PackeeArmTaskStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_ARM_TASK_STATUS__STRUCT_HPP_
