// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from shopee_interfaces:srv/PickeeArmPickProduct.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "shopee_interfaces/srv/pickee_arm_pick_product.hpp"


#ifndef SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_ARM_PICK_PRODUCT__STRUCT_HPP_
#define SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_ARM_PICK_PRODUCT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'target_position'
#include "shopee_interfaces/msg/detail/point3_d__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Request __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Request __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PickeeArmPickProduct_Request_
{
  using Type = PickeeArmPickProduct_Request_<ContainerAllocator>;

  explicit PickeeArmPickProduct_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : target_position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->product_id = 0l;
    }
  }

  explicit PickeeArmPickProduct_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : target_position(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = 0l;
      this->order_id = 0l;
      this->product_id = 0l;
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
  using _target_position_type =
    shopee_interfaces::msg::Point3D_<ContainerAllocator>;
  _target_position_type target_position;

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
  Type & set__target_position(
    const shopee_interfaces::msg::Point3D_<ContainerAllocator> & _arg)
  {
    this->target_position = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Request
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Request
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeArmPickProduct_Request_ & other) const
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
    if (this->target_position != other.target_position) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeArmPickProduct_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeArmPickProduct_Request_

// alias to use template instance with default allocator
using PickeeArmPickProduct_Request =
  shopee_interfaces::srv::PickeeArmPickProduct_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace shopee_interfaces


#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Response __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Response __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PickeeArmPickProduct_Response_
{
  using Type = PickeeArmPickProduct_Response_<ContainerAllocator>;

  explicit PickeeArmPickProduct_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->accepted = false;
      this->message = "";
    }
  }

  explicit PickeeArmPickProduct_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->accepted = false;
      this->message = "";
    }
  }

  // field types and members
  using _accepted_type =
    bool;
  _accepted_type accepted;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;

  // setters for named parameter idiom
  Type & set__accepted(
    const bool & _arg)
  {
    this->accepted = _arg;
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
    shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Response
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Response
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeArmPickProduct_Response_ & other) const
  {
    if (this->accepted != other.accepted) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeArmPickProduct_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeArmPickProduct_Response_

// alias to use template instance with default allocator
using PickeeArmPickProduct_Response =
  shopee_interfaces::srv::PickeeArmPickProduct_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace shopee_interfaces


// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Event __attribute__((deprecated))
#else
# define DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Event __declspec(deprecated)
#endif

namespace shopee_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PickeeArmPickProduct_Event_
{
  using Type = PickeeArmPickProduct_Event_<ContainerAllocator>;

  explicit PickeeArmPickProduct_Event_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : info(_init)
  {
    (void)_init;
  }

  explicit PickeeArmPickProduct_Event_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : info(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _info_type =
    service_msgs::msg::ServiceEventInfo_<ContainerAllocator>;
  _info_type info;
  using _request_type =
    rosidl_runtime_cpp::BoundedVector<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>>>;
  _request_type request;
  using _response_type =
    rosidl_runtime_cpp::BoundedVector<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>>>;
  _response_type response;

  // setters for named parameter idiom
  Type & set__info(
    const service_msgs::msg::ServiceEventInfo_<ContainerAllocator> & _arg)
  {
    this->info = _arg;
    return *this;
  }
  Type & set__request(
    const rosidl_runtime_cpp::BoundedVector<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::srv::PickeeArmPickProduct_Request_<ContainerAllocator>>> & _arg)
  {
    this->request = _arg;
    return *this;
  }
  Type & set__response(
    const rosidl_runtime_cpp::BoundedVector<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<shopee_interfaces::srv::PickeeArmPickProduct_Response_<ContainerAllocator>>> & _arg)
  {
    this->response = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator> *;
  using ConstRawPtr =
    const shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Event
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__shopee_interfaces__srv__PickeeArmPickProduct_Event
    std::shared_ptr<shopee_interfaces::srv::PickeeArmPickProduct_Event_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PickeeArmPickProduct_Event_ & other) const
  {
    if (this->info != other.info) {
      return false;
    }
    if (this->request != other.request) {
      return false;
    }
    if (this->response != other.response) {
      return false;
    }
    return true;
  }
  bool operator!=(const PickeeArmPickProduct_Event_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PickeeArmPickProduct_Event_

// alias to use template instance with default allocator
using PickeeArmPickProduct_Event =
  shopee_interfaces::srv::PickeeArmPickProduct_Event_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace shopee_interfaces

namespace shopee_interfaces
{

namespace srv
{

struct PickeeArmPickProduct
{
  using Request = shopee_interfaces::srv::PickeeArmPickProduct_Request;
  using Response = shopee_interfaces::srv::PickeeArmPickProduct_Response;
  using Event = shopee_interfaces::srv::PickeeArmPickProduct_Event;
};

}  // namespace srv

}  // namespace shopee_interfaces

#endif  // SHOPEE_INTERFACES__SRV__DETAIL__PICKEE_ARM_PICK_PRODUCT__STRUCT_HPP_
