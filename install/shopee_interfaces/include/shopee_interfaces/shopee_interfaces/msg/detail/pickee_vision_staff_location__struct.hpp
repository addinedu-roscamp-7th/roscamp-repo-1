// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PickeeVisionStaffLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_vision_staff_location.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'relative_position'
#include "shopee_interfaces/msg/detail/point2_d__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PickeeVisionStaffLocation __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PickeeVisionStaffLocation __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PickeeVisionStaffLocation_
{
  using Type = PickeeVisionStaffLocation_<ContainerAllocator>;

  explicit PickeeVisionStaffLocation_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : relative_position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->distance = 0.0f;
      this->is_tracking = false;
    }
  }

  explicit PickeeVisionStaffLocation_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : relative_position(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->distance = 0.0f;
      this->is_tracking = false;
    }
  }

  // field types and members
  using _robot_id_type =
    int32_t;
  _robot_id_type robot_id;
  using _relative_position_type =
    shopee_interfaces::msg::Point2D_<ContainerAllocator>;
  _relative_position_type relative_position;
  using _distance_type =
    float;
  _distance_type distance;
  using _is_tracking_type =
    bool;
  _is_tracking_type is_tracking;

  // setters for named parameter idiom
  Type & set__robot_id(
    const int32_t & _arg)
  {
    this->robot_id = _arg;
    return *this;
  }
  Type & set__relative_position(
    const shopee_interfaces::msg::Point2D_<ContainerAllocator> & _arg)
  {
    this->relative_position = _arg;
    return *this;
  }
  Type & set__distance(
    const float & _arg)
  {
    this->distance = _arg;
    return *this;
  }
  Type & set__is_tracking(
    const bool & _arg)
  {
    this->is_tracking = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PickeeVisionStaffLocation
    std::shared_ptr<shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PickeeVisionStaffLocation
    std::shared_ptr<shopee_interfaces::msg::PickeeVisionStaffLocation_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeVisionStaffLocation_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->relative_position != other.relative_position) {
      return false;
    }
    if (this->distance != other.distance) {
      return false;
    }
    if (this->is_tracking != other.is_tracking) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeVisionStaffLocation_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeVisionStaffLocation_

// alias to use template instance with default allocator
using PickeeVisionStaffLocation =
  shopee_interfaces::msg::PickeeVisionStaffLocation_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_VISION_STAFF_LOCATION__STRUCT_HPP_
