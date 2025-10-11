// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from shopee_interfaces:msg/Point2D.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/msg/detail/point2_d__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
shopee_interfaces__msg__Point2D__init(shopee_interfaces__msg__Point2D * msg)
{
  if (!msg) {
    return false;
  }
  // x
  // y
  return true;
}

void
shopee_interfaces__msg__Point2D__fini(shopee_interfaces__msg__Point2D * msg)
{
  if (!msg) {
    return;
  }
  // x
  // y
}

bool
shopee_interfaces__msg__Point2D__are_equal(const shopee_interfaces__msg__Point2D * lhs, const shopee_interfaces__msg__Point2D * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  return true;
}

bool
shopee_interfaces__msg__Point2D__copy(
  const shopee_interfaces__msg__Point2D * input,
  shopee_interfaces__msg__Point2D * output)
{
  if (!input || !output) {
    return false;
  }
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  return true;
}

shopee_interfaces__msg__Point2D *
shopee_interfaces__msg__Point2D__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__Point2D * msg = (shopee_interfaces__msg__Point2D *)allocator.allocate(sizeof(shopee_interfaces__msg__Point2D), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__msg__Point2D));
  bool success = shopee_interfaces__msg__Point2D__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__msg__Point2D__destroy(shopee_interfaces__msg__Point2D * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__msg__Point2D__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__msg__Point2D__Sequence__init(shopee_interfaces__msg__Point2D__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__Point2D * data = NULL;

  if (size) {
    data = (shopee_interfaces__msg__Point2D *)allocator.zero_allocate(size, sizeof(shopee_interfaces__msg__Point2D), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__msg__Point2D__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__msg__Point2D__fini(&data[i - 1]);
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
shopee_interfaces__msg__Point2D__Sequence__fini(shopee_interfaces__msg__Point2D__Sequence * array)
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
      shopee_interfaces__msg__Point2D__fini(&array->data[i]);
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

shopee_interfaces__msg__Point2D__Sequence *
shopee_interfaces__msg__Point2D__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__msg__Point2D__Sequence * array = (shopee_interfaces__msg__Point2D__Sequence *)allocator.allocate(sizeof(shopee_interfaces__msg__Point2D__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__msg__Point2D__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__msg__Point2D__Sequence__destroy(shopee_interfaces__msg__Point2D__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__msg__Point2D__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__msg__Point2D__Sequence__are_equal(const shopee_interfaces__msg__Point2D__Sequence * lhs, const shopee_interfaces__msg__Point2D__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__msg__Point2D__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__msg__Point2D__Sequence__copy(
  const shopee_interfaces__msg__Point2D__Sequence * input,
  shopee_interfaces__msg__Point2D__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__msg__Point2D);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__msg__Point2D * data =
      (shopee_interfaces__msg__Point2D *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__msg__Point2D__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__msg__Point2D__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__msg__Point2D__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
