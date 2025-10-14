// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from shopee_interfaces:srv/MainGetLocationPose.idl
// generated code does not contain a copyright notice
#include "shopee_interfaces/srv/detail/main_get_location_pose__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
shopee_interfaces__srv__MainGetLocationPose_Request__init(shopee_interfaces__srv__MainGetLocationPose_Request * msg)
{
  if (!msg) {
    return false;
  }
  // location_id
  return true;
}

void
shopee_interfaces__srv__MainGetLocationPose_Request__fini(shopee_interfaces__srv__MainGetLocationPose_Request * msg)
{
  if (!msg) {
    return;
  }
  // location_id
}

bool
shopee_interfaces__srv__MainGetLocationPose_Request__are_equal(const shopee_interfaces__srv__MainGetLocationPose_Request * lhs, const shopee_interfaces__srv__MainGetLocationPose_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // location_id
  if (lhs->location_id != rhs->location_id) {
    return false;
  }
  return true;
}

bool
shopee_interfaces__srv__MainGetLocationPose_Request__copy(
  const shopee_interfaces__srv__MainGetLocationPose_Request * input,
  shopee_interfaces__srv__MainGetLocationPose_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // location_id
  output->location_id = input->location_id;
  return true;
}

shopee_interfaces__srv__MainGetLocationPose_Request *
shopee_interfaces__srv__MainGetLocationPose_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Request * msg = (shopee_interfaces__srv__MainGetLocationPose_Request *)allocator.allocate(sizeof(shopee_interfaces__srv__MainGetLocationPose_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__srv__MainGetLocationPose_Request));
  bool success = shopee_interfaces__srv__MainGetLocationPose_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__srv__MainGetLocationPose_Request__destroy(shopee_interfaces__srv__MainGetLocationPose_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__srv__MainGetLocationPose_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__init(shopee_interfaces__srv__MainGetLocationPose_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Request * data = NULL;

  if (size) {
    data = (shopee_interfaces__srv__MainGetLocationPose_Request *)allocator.zero_allocate(size, sizeof(shopee_interfaces__srv__MainGetLocationPose_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__srv__MainGetLocationPose_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__srv__MainGetLocationPose_Request__fini(&data[i - 1]);
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
shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__fini(shopee_interfaces__srv__MainGetLocationPose_Request__Sequence * array)
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
      shopee_interfaces__srv__MainGetLocationPose_Request__fini(&array->data[i]);
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

shopee_interfaces__srv__MainGetLocationPose_Request__Sequence *
shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Request__Sequence * array = (shopee_interfaces__srv__MainGetLocationPose_Request__Sequence *)allocator.allocate(sizeof(shopee_interfaces__srv__MainGetLocationPose_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__destroy(shopee_interfaces__srv__MainGetLocationPose_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__are_equal(const shopee_interfaces__srv__MainGetLocationPose_Request__Sequence * lhs, const shopee_interfaces__srv__MainGetLocationPose_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__srv__MainGetLocationPose_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__copy(
  const shopee_interfaces__srv__MainGetLocationPose_Request__Sequence * input,
  shopee_interfaces__srv__MainGetLocationPose_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__srv__MainGetLocationPose_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__srv__MainGetLocationPose_Request * data =
      (shopee_interfaces__srv__MainGetLocationPose_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__srv__MainGetLocationPose_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__srv__MainGetLocationPose_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__srv__MainGetLocationPose_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `pose`
#include "shopee_interfaces/msg/detail/pose2_d__functions.h"
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

bool
shopee_interfaces__srv__MainGetLocationPose_Response__init(shopee_interfaces__srv__MainGetLocationPose_Response * msg)
{
  if (!msg) {
    return false;
  }
  // pose
  if (!shopee_interfaces__msg__Pose2D__init(&msg->pose)) {
    shopee_interfaces__srv__MainGetLocationPose_Response__fini(msg);
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    shopee_interfaces__srv__MainGetLocationPose_Response__fini(msg);
    return false;
  }
  return true;
}

void
shopee_interfaces__srv__MainGetLocationPose_Response__fini(shopee_interfaces__srv__MainGetLocationPose_Response * msg)
{
  if (!msg) {
    return;
  }
  // pose
  shopee_interfaces__msg__Pose2D__fini(&msg->pose);
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
shopee_interfaces__srv__MainGetLocationPose_Response__are_equal(const shopee_interfaces__srv__MainGetLocationPose_Response * lhs, const shopee_interfaces__srv__MainGetLocationPose_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // pose
  if (!shopee_interfaces__msg__Pose2D__are_equal(
      &(lhs->pose), &(rhs->pose)))
  {
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
shopee_interfaces__srv__MainGetLocationPose_Response__copy(
  const shopee_interfaces__srv__MainGetLocationPose_Response * input,
  shopee_interfaces__srv__MainGetLocationPose_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // pose
  if (!shopee_interfaces__msg__Pose2D__copy(
      &(input->pose), &(output->pose)))
  {
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

shopee_interfaces__srv__MainGetLocationPose_Response *
shopee_interfaces__srv__MainGetLocationPose_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Response * msg = (shopee_interfaces__srv__MainGetLocationPose_Response *)allocator.allocate(sizeof(shopee_interfaces__srv__MainGetLocationPose_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__srv__MainGetLocationPose_Response));
  bool success = shopee_interfaces__srv__MainGetLocationPose_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__srv__MainGetLocationPose_Response__destroy(shopee_interfaces__srv__MainGetLocationPose_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__srv__MainGetLocationPose_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__init(shopee_interfaces__srv__MainGetLocationPose_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Response * data = NULL;

  if (size) {
    data = (shopee_interfaces__srv__MainGetLocationPose_Response *)allocator.zero_allocate(size, sizeof(shopee_interfaces__srv__MainGetLocationPose_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__srv__MainGetLocationPose_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__srv__MainGetLocationPose_Response__fini(&data[i - 1]);
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
shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__fini(shopee_interfaces__srv__MainGetLocationPose_Response__Sequence * array)
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
      shopee_interfaces__srv__MainGetLocationPose_Response__fini(&array->data[i]);
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

shopee_interfaces__srv__MainGetLocationPose_Response__Sequence *
shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Response__Sequence * array = (shopee_interfaces__srv__MainGetLocationPose_Response__Sequence *)allocator.allocate(sizeof(shopee_interfaces__srv__MainGetLocationPose_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__destroy(shopee_interfaces__srv__MainGetLocationPose_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__are_equal(const shopee_interfaces__srv__MainGetLocationPose_Response__Sequence * lhs, const shopee_interfaces__srv__MainGetLocationPose_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__srv__MainGetLocationPose_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__copy(
  const shopee_interfaces__srv__MainGetLocationPose_Response__Sequence * input,
  shopee_interfaces__srv__MainGetLocationPose_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__srv__MainGetLocationPose_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__srv__MainGetLocationPose_Response * data =
      (shopee_interfaces__srv__MainGetLocationPose_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__srv__MainGetLocationPose_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__srv__MainGetLocationPose_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__srv__MainGetLocationPose_Response__copy(
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
// #include "shopee_interfaces/srv/detail/main_get_location_pose__functions.h"

bool
shopee_interfaces__srv__MainGetLocationPose_Event__init(shopee_interfaces__srv__MainGetLocationPose_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    shopee_interfaces__srv__MainGetLocationPose_Event__fini(msg);
    return false;
  }
  // request
  if (!shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__init(&msg->request, 0)) {
    shopee_interfaces__srv__MainGetLocationPose_Event__fini(msg);
    return false;
  }
  // response
  if (!shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__init(&msg->response, 0)) {
    shopee_interfaces__srv__MainGetLocationPose_Event__fini(msg);
    return false;
  }
  return true;
}

void
shopee_interfaces__srv__MainGetLocationPose_Event__fini(shopee_interfaces__srv__MainGetLocationPose_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__fini(&msg->request);
  // response
  shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__fini(&msg->response);
}

bool
shopee_interfaces__srv__MainGetLocationPose_Event__are_equal(const shopee_interfaces__srv__MainGetLocationPose_Event * lhs, const shopee_interfaces__srv__MainGetLocationPose_Event * rhs)
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
  if (!shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
shopee_interfaces__srv__MainGetLocationPose_Event__copy(
  const shopee_interfaces__srv__MainGetLocationPose_Event * input,
  shopee_interfaces__srv__MainGetLocationPose_Event * output)
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
  if (!shopee_interfaces__srv__MainGetLocationPose_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!shopee_interfaces__srv__MainGetLocationPose_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

shopee_interfaces__srv__MainGetLocationPose_Event *
shopee_interfaces__srv__MainGetLocationPose_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Event * msg = (shopee_interfaces__srv__MainGetLocationPose_Event *)allocator.allocate(sizeof(shopee_interfaces__srv__MainGetLocationPose_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(shopee_interfaces__srv__MainGetLocationPose_Event));
  bool success = shopee_interfaces__srv__MainGetLocationPose_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
shopee_interfaces__srv__MainGetLocationPose_Event__destroy(shopee_interfaces__srv__MainGetLocationPose_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    shopee_interfaces__srv__MainGetLocationPose_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
shopee_interfaces__srv__MainGetLocationPose_Event__Sequence__init(shopee_interfaces__srv__MainGetLocationPose_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Event * data = NULL;

  if (size) {
    data = (shopee_interfaces__srv__MainGetLocationPose_Event *)allocator.zero_allocate(size, sizeof(shopee_interfaces__srv__MainGetLocationPose_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = shopee_interfaces__srv__MainGetLocationPose_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        shopee_interfaces__srv__MainGetLocationPose_Event__fini(&data[i - 1]);
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
shopee_interfaces__srv__MainGetLocationPose_Event__Sequence__fini(shopee_interfaces__srv__MainGetLocationPose_Event__Sequence * array)
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
      shopee_interfaces__srv__MainGetLocationPose_Event__fini(&array->data[i]);
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

shopee_interfaces__srv__MainGetLocationPose_Event__Sequence *
shopee_interfaces__srv__MainGetLocationPose_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  shopee_interfaces__srv__MainGetLocationPose_Event__Sequence * array = (shopee_interfaces__srv__MainGetLocationPose_Event__Sequence *)allocator.allocate(sizeof(shopee_interfaces__srv__MainGetLocationPose_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = shopee_interfaces__srv__MainGetLocationPose_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
shopee_interfaces__srv__MainGetLocationPose_Event__Sequence__destroy(shopee_interfaces__srv__MainGetLocationPose_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    shopee_interfaces__srv__MainGetLocationPose_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
shopee_interfaces__srv__MainGetLocationPose_Event__Sequence__are_equal(const shopee_interfaces__srv__MainGetLocationPose_Event__Sequence * lhs, const shopee_interfaces__srv__MainGetLocationPose_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!shopee_interfaces__srv__MainGetLocationPose_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
shopee_interfaces__srv__MainGetLocationPose_Event__Sequence__copy(
  const shopee_interfaces__srv__MainGetLocationPose_Event__Sequence * input,
  shopee_interfaces__srv__MainGetLocationPose_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(shopee_interfaces__srv__MainGetLocationPose_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    shopee_interfaces__srv__MainGetLocationPose_Event * data =
      (shopee_interfaces__srv__MainGetLocationPose_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!shopee_interfaces__srv__MainGetLocationPose_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          shopee_interfaces__srv__MainGetLocationPose_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!shopee_interfaces__srv__MainGetLocationPose_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
