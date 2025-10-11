// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/Obstacle.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/obstacle.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'position'
#include "shopee_interfaces/msg/detail/point2_d__struct.hpp"
// Member 'direction'
#include "shopee_interfaces/msg/detail/vector2_d__struct.hpp"
// Member 'bbox'
#include "shopee_interfaces/msg/detail/b_box__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__Obstacle __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__Obstacle __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Obstacle_
{
  using Type = Obstacle_<ContainerAllocator>;

  explicit Obstacle_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init),
    direction(_init),
    bbox(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->obstacle_type = "";
      this->distance = 0.0f;
      this->velocity = 0.0f;
      this->confidence = 0.0f;
    }
  }

  explicit Obstacle_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : obstacle_type(_alloc),
    position(_alloc, _init),
    direction(_alloc, _init),
    bbox(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->obstacle_type = "";
      this->distance = 0.0f;
      this->velocity = 0.0f;
      this->confidence = 0.0f;
    }
  }

  // field types and members
  using _obstacle_type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _obstacle_type_type obstacle_type;
  using _position_type =
    shopee_interfaces::msg::Point2D_<ContainerAllocator>;
  _position_type position;
  using _distance_type =
    float;
  _distance_type distance;
  using _velocity_type =
    float;
  _velocity_type velocity;
  using _direction_type =
    shopee_interfaces::msg::Vector2D_<ContainerAllocator>;
  _direction_type direction;
  using _bbox_type =
    shopee_interfaces::msg::BBox_<ContainerAllocator>;
  _bbox_type bbox;
  using _confidence_type =
    float;
  _confidence_type confidence;

  // setters for named parameter idiom
  Type & set__obstacle_type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->obstacle_type = _arg;
    return *this;
  }
  Type & set__position(
    const shopee_interfaces::msg::Point2D_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__distance(
    const float & _arg)
  {
    this->distance = _arg;
    return *this;
  }
  Type & set__velocity(
    const float & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__direction(
    const shopee_interfaces::msg::Vector2D_<ContainerAllocator> & _arg)
  {
    this->direction = _arg;
    return *this;
  }
  Type & set__bbox(
    const shopee_interfaces::msg::BBox_<ContainerAllocator> & _arg)
  {
    this->bbox = _arg;
    return *this;
  }
  Type & set__confidence(
    const float & _arg)
  {
    this->confidence = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::Obstacle_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::Obstacle_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::Obstacle_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::Obstacle_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::Obstacle_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::Obstacle_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::Obstacle_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::Obstacle_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::Obstacle_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::Obstacle_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__Obstacle
    std::shared_ptr<shopee_interfaces::msg::Obstacle_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__Obstacle
    std::shared_ptr<shopee_interfaces::msg::Obstacle_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Obstacle_ & other) const
  {
    if (this->obstacle_type != other.obstacle_type) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->distance != other.distance) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->direction != other.direction) {
      return false;
    }
    if (this->bbox != other.bbox) {
      return false;
    }
    if (this->confidence != other.confidence) {
      return false;
    }
    return true;
  }
  bool operator!=(const Obstacle_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Obstacle_

// alias to use template instance with default allocator
using Obstacle =
  shopee_interfaces::msg::Obstacle_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__OBSTACLE__STRUCT_HPP_
