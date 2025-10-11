// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PickeeDetectedProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_detected_product.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'bbox_coords'
#include "shopee_interfaces/msg/detail/b_box__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PickeeDetectedProduct __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PickeeDetectedProduct __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PickeeDetectedProduct_
{
  using Type = PickeeDetectedProduct_<ContainerAllocator>;

  explicit PickeeDetectedProduct_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : bbox_coords(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->product_id = 0l;
      this->bbox_number = 0l;
      this->confidence = 0.0f;
    }
  }

  explicit PickeeDetectedProduct_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : bbox_coords(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->product_id = 0l;
      this->bbox_number = 0l;
      this->confidence = 0.0f;
    }
  }

  // field types and members
  using _product_id_type =
    int32_t;
  _product_id_type product_id;
  using _bbox_number_type =
    int32_t;
  _bbox_number_type bbox_number;
  using _bbox_coords_type =
    shopee_interfaces::msg::BBox_<ContainerAllocator>;
  _bbox_coords_type bbox_coords;
  using _confidence_type =
    float;
  _confidence_type confidence;

  // setters for named parameter idiom
  Type & set__product_id(
    const int32_t & _arg)
  {
    this->product_id = _arg;
    return *this;
  }
  Type & set__bbox_number(
    const int32_t & _arg)
  {
    this->bbox_number = _arg;
    return *this;
  }
  Type & set__bbox_coords(
    const shopee_interfaces::msg::BBox_<ContainerAllocator> & _arg)
  {
    this->bbox_coords = _arg;
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
    shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PickeeDetectedProduct
    std::shared_ptr<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PickeeDetectedProduct
    std::shared_ptr<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeDetectedProduct_ & other) const
  {
    if (this->product_id != other.product_id) {
      return false;
    }
    if (this->bbox_number != other.bbox_number) {
      return false;
    }
    if (this->bbox_coords != other.bbox_coords) {
      return false;
    }
    if (this->confidence != other.confidence) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeDetectedProduct_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeDetectedProduct_

// alias to use template instance with default allocator
using PickeeDetectedProduct =
  shopee_interfaces::msg::PickeeDetectedProduct_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_DETECTED_PRODUCT__STRUCT_HPP_
