// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from shopee_interfaces:msg/PickeeMoveStatus.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/msg/detail/pickee_move_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
shopee_interfaces__msg__PickeeMoveStatus__init(shopee_interfaces__msg__PickeeMoveStatus * msg)
{
  if (!msg) {
    return false;
  }
  // robot_id
  // order_id
  // location_id
  return true;
}

void
shopee_interfaces__msg__PickeeMoveStatus__fini(shopee_interfaces__msg__PickeeMoveStatus * msg)
{
  if (!msg) {
    return;
  }
  // robot_id
  // order_id
  // location_id
}

bool
shopee_interfaces__msg__PickeeMoveStatus__are_equal(const shopee_interfaces__msg__PickeeMoveStatus * lhs, const shopee_interfaces__msg__PickeeMoveStatus * rhs)
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
  // location_id
  if (lhs->location_id != rhs->location_id) {
    return false;
  }
  return true;
}

bool
shopee_interfaces__msg__PickeeMoveStatus__copy(
  const shopee_interfaces__msg__PickeeMoveStatus * input,
  shopee_interfaces__msg__PickeeMoveStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // robot_id
  output->robot_id = input->robot_id;
  // order_id
  output->order_id = input->order_id;
  // location_id
  output->location_id = input->location_id;
  return true;
}

shopee_interfaces__msg__PickeeMoveStatus *
shopee_interfaces__msg__PickeeMoveStatus__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__PickeeMoveStatus * msg = (shopee_interfaces__msg__PickeeMoveStatus *)allocator.allocate(sizeof(shopee_interfaces__msg__PickeeMoveStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__msg__PickeeMoveStatus));
  bool success = shopee_interfaces__msg__PickeeMoveStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__msg__PickeeMoveStatus__destroy(shopee_interfaces__msg__PickeeMoveStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__msg__PickeeMoveStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__msg__PickeeMoveStatus__Sequence__init(shopee_interfaces__msg__PickeeMoveStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__PickeeMoveStatus * data = NULL;

  if (size) {
    data = (shopee_interfaces__msg__PickeeMoveStatus *)allocator.zero_allocate(size, sizeof(shopee_interfaces__msg__PickeeMoveStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__msg__PickeeMoveStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__msg__PickeeMoveStatus__fini(&data[i - 1]);
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
shopee_interfaces__msg__PickeeMoveStatus__Sequence__fini(shopee_interfaces__msg__PickeeMoveStatus__Sequence * array)
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
      shopee_interfaces__msg__PickeeMoveStatus__fini(&array->data[i]);
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

shopee_interfaces__msg__PickeeMoveStatus__Sequence *
shopee_interfaces__msg__PickeeMoveStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__PickeeMoveStatus__Sequence * array = (shopee_interfaces__msg__PickeeMoveStatus__Sequence *)allocator.allocate(sizeof(shopee_interfaces__msg__PickeeMoveStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__msg__PickeeMoveStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__msg__PickeeMoveStatus__Sequence__destroy(shopee_interfaces__msg__PickeeMoveStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__msg__PickeeMoveStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__msg__PickeeMoveStatus__Sequence__are_equal(const shopee_interfaces__msg__PickeeMoveStatus__Sequence * lhs, const shopee_interfaces__msg__PickeeMoveStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__msg__PickeeMoveStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__msg__PickeeMoveStatus__Sequence__copy(
  const shopee_interfaces__msg__PickeeMoveStatus__Sequence * input,
  shopee_interfaces__msg__PickeeMoveStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__msg__PickeeMoveStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__msg__PickeeMoveStatus * data =
      (shopee_interfaces__msg__PickeeMoveStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__msg__PickeeMoveStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__msg__PickeeMoveStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__msg__PickeeMoveStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
