// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/Vector2D.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/vector2_d.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__Vector2D __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__Vector2D __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Vector2D_
{
  using Type = Vector2D_<ContainerAllocator>;

  explicit Vector2D_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->vx = 0.0f;
      this->vy = 0.0f;
    }
  }

  explicit Vector2D_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->vx = 0.0f;
      this->vy = 0.0f;
    }
  }

  // field types and members
  using _vx_type =
    float;
  _vx_type vx;
  using _vy_type =
    float;
  _vy_type vy;

  // setters for named parameter idiom
  Type & set__vx(
    const float & _arg)
  {
    this->vx = _arg;
    return *this;
  }
  Type & set__vy(
    const float & _arg)
  {
    this->vy = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::Vector2D_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::Vector2D_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::Vector2D_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::Vector2D_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::Vector2D_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::Vector2D_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::Vector2D_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::Vector2D_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::Vector2D_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::Vector2D_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__Vector2D
    std::shared_ptr<shopee_interfaces::msg::Vector2D_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__Vector2D
    std::shared_ptr<shopee_interfaces::msg::Vector2D_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Vector2D_ & other) const
  {
    if (this->vx != other.vx) {
      return false;
    }
    if (this->vy != other.vy) {
      return false;
    }
    return true;
  }
  bool operator!=(const Vector2D_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Vector2D_

// alias to use template instance with default allocator
using Vector2D =
  shopee_interfaces::msg::Vector2D_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__VECTOR2_D__STRUCT_HPP_
