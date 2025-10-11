// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from shopee_interfaces:msg/PackeeArmTaskStatus.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/msg/detail/packee_arm_task_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `arm_side`
// Member `status`
// Member `current_phase`
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

bool
shopee_interfaces__msg__PackeeArmTaskStatus__init(shopee_interfaces__msg__PackeeArmTaskStatus * msg)
{
  if (!msg) {
    return false;
  }
  // robot_id
  // order_id
  // product_id
  // arm_side
  if (!rosidl_runtime_c__String__init(&msg->arm_side)) {
    shopee_interfaces__msg__PackeeArmTaskStatus__fini(msg);
    return false;
  }
  // status
  if (!rosidl_runtime_c__String__init(&msg->status)) {
    shopee_interfaces__msg__PackeeArmTaskStatus__fini(msg);
    return false;
  }
  // current_phase
  if (!rosidl_runtime_c__String__init(&msg->current_phase)) {
    shopee_interfaces__msg__PackeeArmTaskStatus__fini(msg);
    return false;
  }
  // progress
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    shopee_interfaces__msg__PackeeArmTaskStatus__fini(msg);
    return false;
  }
  return true;
}

void
shopee_interfaces__msg__PackeeArmTaskStatus__fini(shopee_interfaces__msg__PackeeArmTaskStatus * msg)
{
  if (!msg) {
    return;
  }
  // robot_id
  // order_id
  // product_id
  // arm_side
  rosidl_runtime_c__String__fini(&msg->arm_side);
  // status
  rosidl_runtime_c__String__fini(&msg->status);
  // current_phase
  rosidl_runtime_c__String__fini(&msg->current_phase);
  // progress
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
shopee_interfaces__msg__PackeeArmTaskStatus__are_equal(const shopee_interfaces__msg__PackeeArmTaskStatus * lhs, const shopee_interfaces__msg__PackeeArmTaskStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // robot_id
  if (lhs->robot_id != rhs->robot_id) {
    return false;
  }
  // order_id
  if (lhs->order_id != rhs->order_id) {
    return false;
  }
  // product_id
  if (lhs->product_id != rhs->product_id) {
    return false;
  }
  // arm_side
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->arm_side), &(rhs->arm_side)))
  {
    return false;
  }
  // status
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->status), &(rhs->status)))
  {
    return false;
  }
  // current_phase
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->current_phase), &(rhs->current_phase)))
  {
    return false;
  }
  // progress
  if (lhs->progress != rhs->progress) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  return true;
}

bool
shopee_interfaces__msg__PackeeArmTaskStatus__copy(
  const shopee_interfaces__msg__PackeeArmTaskStatus * input,
  shopee_interfaces__msg__PackeeArmTaskStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // robot_id
  output->robot_id = input->robot_id;
  // order_id
  output->order_id = input->order_id;
  // product_id
  output->product_id = input->product_id;
  // arm_side
  if (!rosidl_runtime_c__String__copy(
      &(input->arm_side), &(output->arm_side)))
  {
    return false;
  }
  // status
  if (!rosidl_runtime_c__String__copy(
      &(input->status), &(output->status)))
  {
    return false;
  }
  // current_phase
  if (!rosidl_runtime_c__String__copy(
      &(input->current_phase), &(output->current_phase)))
  {
    return false;
  }
  // progress
  output->progress = input->progress;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  return true;
}

shopee_interfaces__msg__PackeeArmTaskStatus *
shopee_interfaces__msg__PackeeArmTaskStatus__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__PackeeArmTaskStatus * msg = (shopee_interfaces__msg__PackeeArmTaskStatus *)allocator.allocate(sizeof(shopee_interfaces__msg__PackeeArmTaskStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__msg__PackeeArmTaskStatus));
  bool success = shopee_interfaces__msg__PackeeArmTaskStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__msg__PackeeArmTaskStatus__destroy(shopee_interfaces__msg__PackeeArmTaskStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__msg__PackeeArmTaskStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__msg__PackeeArmTaskStatus__Sequence__init(shopee_interfaces__msg__PackeeArmTaskStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__PackeeArmTaskStatus * data = NULL;

  if (size) {
    data = (shopee_interfaces__msg__PackeeArmTaskStatus *)allocator.zero_allocate(size, sizeof(shopee_interfaces__msg__PackeeArmTaskStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__msg__PackeeArmTaskStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__msg__PackeeArmTaskStatus__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
shopee_interfaces__msg__PackeeArmTaskStatus__Sequence__fini(shopee_interfaces__msg__PackeeArmTaskStatus__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      shopee_interfaces__msg__PackeeArmTaskStatus__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

shopee_interfaces__msg__PackeeArmTaskStatus__Sequence *
shopee_interfaces__msg__PackeeArmTaskStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__PackeeArmTaskStatus__Sequence * array = (shopee_interfaces__msg__PackeeArmTaskStatus__Sequence *)allocator.allocate(sizeof(shopee_interfaces__msg__PackeeArmTaskStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__msg__PackeeArmTaskStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__msg__PackeeArmTaskStatus__Sequence__destroy(shopee_interfaces__msg__PackeeArmTaskStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__msg__PackeeArmTaskStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__msg__PackeeArmTaskStatus__Sequence__are_equal(const shopee_interfaces__msg__PackeeArmTaskStatus__Sequence * lhs, const shopee_interfaces__msg__PackeeArmTaskStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__msg__PackeeArmTaskStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__msg__PackeeArmTaskStatus__Sequence__copy(
  const shopee_interfaces__msg__PackeeArmTaskStatus__Sequence * input,
  shopee_interfaces__msg__PackeeArmTaskStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__msg__PackeeArmTaskStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__msg__PackeeArmTaskStatus * data =
      (shopee_interfaces__msg__PackeeArmTaskStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__msg__PackeeArmTaskStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__msg__PackeeArmTaskStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__msg__PackeeArmTaskStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
