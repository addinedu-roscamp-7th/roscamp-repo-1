// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PickeeVisionCartCheck.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_cart_check.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_CART_CHECK__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_CART_CHECK__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PickeeVisionCartCheck __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PickeeVisionCartCheck __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PickeeVisionCartCheck_
{
  using Type = PickeeVisionCartCheck_<ContainerAllocator>;

  explicit PickeeVisionCartCheck_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->success = false;
      this->product_id = 0l;
      this->found = false;
      this->quantity = 0l;
      this->message = "";
    }
  }

  explicit PickeeVisionCartCheck_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->success = false;
      this->product_id = 0l;
      this->found = false;
      this->quantity = 0l;
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
  using _success_type =
    bool;
  _success_type success;
  using _product_id_type =
    int32_t;
  _product_id_type product_id;
  using _found_type =
    bool;
  _found_type found;
  using _quantity_type =
    int32_t;
  _quantity_type quantity;
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
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__product_id(
    const int32_t & _arg)
  {
    this->product_id = _arg;
    return *this;
  }
  Type & set__found(
    const bool & _arg)
  {
    this->found = _arg;
    return *this;
  }
  Type & set__quantity(
    const int32_t & _arg)
  {
    this->quantity = _arg;
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
    shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PickeeVisionCartCheck
    std::shared_ptr<shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PickeeVisionCartCheck
    std::shared_ptr<shopee_interfaces::msg::PickeeVisionCartCheck_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeVisionCartCheck_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->order_id != other.order_id) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    if (this->product_id != other.product_id) {
      return false;
    }
    if (this->found != other.found) {
      return false;
    }
    if (this->quantity != other.quantity) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeVisionCartCheck_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeVisionCartCheck_

// alias to use template instance with default allocator
using PickeeVisionCartCheck =
  shopee_interfaces::msg::PickeeVisionCartCheck_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_CART_CHECK__STRUCT_HPP_
