// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:msg/ProductLocation.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/msg/product_location.hpp"


#ifndef SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__STRUCT_HPP_
#define SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__msg__ProductLocation __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__msg__ProductLocation __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ProductLocation_
{
  using Type = ProductLocation_<ContainerAllocator>;

  explicit ProductLocation_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->product_id = 0l;
      this->location_id = 0l;
      this->section_id = 0l;
      this->quantity = 0l;
    }
  }

  explicit ProductLocation_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->product_id = 0l;
      this->location_id = 0l;
      this->section_id = 0l;
      this->quantity = 0l;
    }
  }

  // field types and members
  using _product_id_type =
    int32_t;
  _product_id_type product_id;
  using _location_id_type =
    int32_t;
  _location_id_type location_id;
  using _section_id_type =
    int32_t;
  _section_id_type section_id;
  using _quantity_type =
    int32_t;
  _quantity_type quantity;

  // setters for named parameter idiom
  Type & set__product_id(
    const int32_t & _arg)
  {
    this->product_id = _arg;
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
  Type & set__quantity(
    const int32_t & _arg)
  {
    this->quantity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::msg::ProductLocation_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::msg::ProductLocation_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::msg::ProductLocation_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::msg::ProductLocation_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::ProductLocation_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::ProductLocation_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::msg::ProductLocation_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::msg::ProductLocation_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::msg::ProductLocation_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::msg::ProductLocation_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__msg__ProductLocation
    std::shared_ptr<shopee_interfaces::msg::ProductLocation_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__msg__ProductLocation
    std::shared_ptr<shopee_interfaces::msg::ProductLocation_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ProductLocation_ & other) const
  {
    if (this->product_id != other.product_id) {
      return false;
    }
    if (this->location_id != other.location_id) {
      return false;
    }
    if (this->section_id != other.section_id) {
      return false;
    }
    if (this->quantity != other.quantity) {
      return false;
    }
    return true;
  }
  bool operator!=(const ProductLocation_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ProductLocation_

// alias to use template instance with default allocator
using ProductLocation =
  shopee_interfaces::msg::ProductLocation_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__MSG__DETAIL__PRODUCT_LOCATION__STRUCT_HPP_
