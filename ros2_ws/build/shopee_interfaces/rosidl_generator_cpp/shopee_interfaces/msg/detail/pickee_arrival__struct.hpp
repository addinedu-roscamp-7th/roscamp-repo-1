// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PickeeArrival.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_arrival.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARRIVAL__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARRIVAL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PickeeArrival __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PickeeArrival __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PickeeArrival_
{
  using Type = PickeeArrival_<ContainerAllocator>;

  explicit PickeeArrival_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->location_id = 0l;
      this->section_id = 0l;
    }
  }

  explicit PickeeArrival_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->location_id = 0l;
      this->section_id = 0l;
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
  using _section_id_type =
    int32_t;
  _section_id_type section_id;

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
  Type & set__section_id(
    const int32_t & _arg)
  {
    this->section_id = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::PickeeArrival_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PickeeArrival_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeArrival_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeArrival_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeArrival_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeArrival_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeArrival_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeArrival_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeArrival_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeArrival_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PickeeArrival
    std::shared_ptr<shopee_interfaces::msg::PickeeArrival_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PickeeArrival
    std::shared_ptr<shopee_interfaces::msg::PickeeArrival_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeArrival_ & other) const
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
    if (this->section_id != other.section_id) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeArrival_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeArrival_

// alias to use template instance with default allocator
using PickeeArrival =
  shopee_interfaces::msg::PickeeArrival_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_ARRIVAL__STRUCT_HPP_
