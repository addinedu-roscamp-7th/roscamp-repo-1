// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from shopee_interfaces:srv/PickeeProductProcessSelection.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/srv/detail/pickee_product_process_selection__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Request__init(shopee_interfaces__srv__PickeeProductProcessSelection_Request * msg)
{
  if (!msg) {
    return false;
  }
  // robot_id
  // order_id
  // product_id
  // bbox_number
  return true;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Request__fini(shopee_interfaces__srv__PickeeProductProcessSelection_Request * msg)
{
  if (!msg) {
    return;
  }
  // robot_id
  // order_id
  // product_id
  // bbox_number
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Request__are_equal(const shopee_interfaces__srv__PickeeProductProcessSelection_Request * lhs, const shopee_interfaces__srv__PickeeProductProcessSelection_Request * rhs)
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
  // bbox_number
  if (lhs->bbox_number != rhs->bbox_number) {
    return false;
  }
  return true;
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Request__copy(
  const shopee_interfaces__srv__PickeeProductProcessSelection_Request * input,
  shopee_interfaces__srv__PickeeProductProcessSelection_Request * output)
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
  // bbox_number
  output->bbox_number = input->bbox_number;
  return true;
}

shopee_interfaces__srv__PickeeProductProcessSelection_Request *
shopee_interfaces__srv__PickeeProductProcessSelection_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Request * msg = (shopee_interfaces__srv__PickeeProductProcessSelection_Request *)allocator.allocate(sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Request));
  bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Request__destroy(shopee_interfaces__srv__PickeeProductProcessSelection_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__init(shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Request * data = NULL;

  if (size) {
    data = (shopee_interfaces__srv__PickeeProductProcessSelection_Request *)allocator.zero_allocate(size, sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__srv__PickeeProductProcessSelection_Request__fini(&data[i - 1]);
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
shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__fini(shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence * array)
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
      shopee_interfaces__srv__PickeeProductProcessSelection_Request__fini(&array->data[i]);
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

shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence *
shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence * array = (shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence *)allocator.allocate(sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__destroy(shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__are_equal(const shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence * lhs, const shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__srv__PickeeProductProcessSelection_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__copy(
  const shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence * input,
  shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__srv__PickeeProductProcessSelection_Request * data =
      (shopee_interfaces__srv__PickeeProductProcessSelection_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__srv__PickeeProductProcessSelection_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__srv__PickeeProductProcessSelection_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__srv__PickeeProductProcessSelection_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Response__init(shopee_interfaces__srv__PickeeProductProcessSelection_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Response__fini(msg);
    return false;
  }
  return true;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Response__fini(shopee_interfaces__srv__PickeeProductProcessSelection_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Response__are_equal(const shopee_interfaces__srv__PickeeProductProcessSelection_Response * lhs, const shopee_interfaces__srv__PickeeProductProcessSelection_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
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
shopee_interfaces__srv__PickeeProductProcessSelection_Response__copy(
  const shopee_interfaces__srv__PickeeProductProcessSelection_Response * input,
  shopee_interfaces__srv__PickeeProductProcessSelection_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  return true;
}

shopee_interfaces__srv__PickeeProductProcessSelection_Response *
shopee_interfaces__srv__PickeeProductProcessSelection_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Response * msg = (shopee_interfaces__srv__PickeeProductProcessSelection_Response *)allocator.allocate(sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Response));
  bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Response__destroy(shopee_interfaces__srv__PickeeProductProcessSelection_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__init(shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Response * data = NULL;

  if (size) {
    data = (shopee_interfaces__srv__PickeeProductProcessSelection_Response *)allocator.zero_allocate(size, sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__srv__PickeeProductProcessSelection_Response__fini(&data[i - 1]);
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
shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__fini(shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence * array)
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
      shopee_interfaces__srv__PickeeProductProcessSelection_Response__fini(&array->data[i]);
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

shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence *
shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence * array = (shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence *)allocator.allocate(sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__destroy(shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__are_equal(const shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence * lhs, const shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__srv__PickeeProductProcessSelection_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__copy(
  const shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence * input,
  shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__srv__PickeeProductProcessSelection_Response * data =
      (shopee_interfaces__srv__PickeeProductProcessSelection_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__srv__PickeeProductProcessSelection_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__srv__PickeeProductProcessSelection_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__srv__PickeeProductProcessSelection_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `info`
#include "service_msgs/msg/detail/service_event_info__functions.h"
// Member `request`
// Member `response`
// already included above
// #include "shopee_interfaces/srv/detail/pickee_product_process_selection__functions.h"

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Event__init(shopee_interfaces__srv__PickeeProductProcessSelection_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Event__fini(msg);
    return false;
  }
  // request
  if (!shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__init(&msg->request, 0)) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Event__fini(msg);
    return false;
  }
  // response
  if (!shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__init(&msg->response, 0)) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Event__fini(msg);
    return false;
  }
  return true;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Event__fini(shopee_interfaces__srv__PickeeProductProcessSelection_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__fini(&msg->request);
  // response
  shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__fini(&msg->response);
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Event__are_equal(const shopee_interfaces__srv__PickeeProductProcessSelection_Event * lhs, const shopee_interfaces__srv__PickeeProductProcessSelection_Event * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__are_equal(
      &(lhs->info), &(rhs->info)))
  {
    return false;
  }
  // request
  if (!shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Event__copy(
  const shopee_interfaces__srv__PickeeProductProcessSelection_Event * input,
  shopee_interfaces__srv__PickeeProductProcessSelection_Event * output)
{
  if (!input || !output) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__copy(
      &(input->info), &(output->info)))
  {
    return false;
  }
  // request
  if (!shopee_interfaces__srv__PickeeProductProcessSelection_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!shopee_interfaces__srv__PickeeProductProcessSelection_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

shopee_interfaces__srv__PickeeProductProcessSelection_Event *
shopee_interfaces__srv__PickeeProductProcessSelection_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Event * msg = (shopee_interfaces__srv__PickeeProductProcessSelection_Event *)allocator.allocate(sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Event));
  bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Event__destroy(shopee_interfaces__srv__PickeeProductProcessSelection_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence__init(shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Event * data = NULL;

  if (size) {
    data = (shopee_interfaces__srv__PickeeProductProcessSelection_Event *)allocator.zero_allocate(size, sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__srv__PickeeProductProcessSelection_Event__fini(&data[i - 1]);
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
shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence__fini(shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence * array)
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
      shopee_interfaces__srv__PickeeProductProcessSelection_Event__fini(&array->data[i]);
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

shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence *
shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence * array = (shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence *)allocator.allocate(sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence__destroy(shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence__are_equal(const shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence * lhs, const shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__srv__PickeeProductProcessSelection_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence__copy(
  const shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence * input,
  shopee_interfaces__srv__PickeeProductProcessSelection_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__srv__PickeeProductProcessSelection_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__srv__PickeeProductProcessSelection_Event * data =
      (shopee_interfaces__srv__PickeeProductProcessSelection_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__srv__PickeeProductProcessSelection_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__srv__PickeeProductProcessSelection_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__srv__PickeeProductProcessSelection_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
