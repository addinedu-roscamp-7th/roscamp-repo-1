// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/PickeeProductDetection.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/pickee_product_detection.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_DETECTION__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_DETECTION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'products'
#include "shopee_interfaces/msg/detail/pickee_detected_product__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__PickeeProductDetection __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__PickeeProductDetection __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PickeeProductDetection_
{
  using Type = PickeeProductDetection_<ContainerAllocator>;

  explicit PickeeProductDetection_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
    }
  }

  explicit PickeeProductDetection_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
    }
  }

  // field types and members
  using _robot_id_type =
    int32_t;
  _robot_id_type robot_id;
  using _order_id_type =
    int32_t;
  _order_id_type order_id;
  using _products_type =
    std::vector<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>>>;
  _products_type products;

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
  Type & set__products(
    const std::vector<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::msg::PickeeDetectedProduct_<ContainerAllocator>>> & _arg)
  {
    this->products = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__PickeeProductDetection
    std::shared_ptr<shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__PickeeProductDetection
    std::shared_ptr<shopee_interfaces::msg::PickeeProductDetection_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeProductDetection_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->order_id != other.order_id) {
      return false;
    }
    if (this->products != other.products) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeProductDetection_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeProductDetection_

// alias to use template instance with default allocator
using PickeeProductDetection =
  shopee_interfaces::msg::PickeeProductDetection_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PICKEE_PRODUCT_DETECTION__STRUCT_HPP_
