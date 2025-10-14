// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PackeeDetectedProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/packee_detected_product.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'bbox'
#include "shopee_interfaces/msg/detail/b_box__struct.hpp"
// Member 'position'
#include "shopee_interfaces/msg/detail/point3_d__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PackeeDetectedProduct __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PackeeDetectedProduct __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PackeeDetectedProduct_
{
  using Type = PackeeDetectedProduct_<ContainerAllocator>;

  explicit PackeeDetectedProduct_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : bbox(_init),
    position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->product_id = 0l;
      this->confidence = 0.0f;
    }
  }

  explicit PackeeDetectedProduct_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : bbox(_alloc, _init),
    position(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->product_id = 0l;
      this->confidence = 0.0f;
    }
  }

  // field types and members
  using _product_id_type =
    int32_t;
  _product_id_type product_id;
  using _bbox_type =
    shopee_interfaces::msg::BBox_<ContainerAllocator>;
  _bbox_type bbox;
  using _confidence_type =
    float;
  _confidence_type confidence;
  using _position_type =
    shopee_interfaces::msg::Point3D_<ContainerAllocator>;
  _position_type position;

  // setters for named parameter idiom
  Type & set__product_id(
    const int32_t & _arg)
  {
    this->product_id = _arg;
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
  Type & set__position(
    const shopee_interfaces::msg::Point3D_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PackeeDetectedProduct
    std::shared_ptr<shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PackeeDetectedProduct
    std::shared_ptr<shopee_interfaces::msg::PackeeDetectedProduct_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PackeeDetectedProduct_ & other) const
  {
    if (this->product_id != other.product_id) {
      return false;
    }
    if (this->bbox != other.bbox) {
      return false;
    }
    if (this->confidence != other.confidence) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    return true;
  }
  bool operator!=(const PackeeDetectedProduct_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PackeeDetectedProduct_

// alias to use template instance with default allocator
using PackeeDetectedProduct =
  shopee_interfaces::msg::PackeeDetectedProduct_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PACKEE_DETECTED_PRODUCT__STRUCT_HPP_
