// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from shopee_interfaces:msg/PickeeMobileArrival.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "shopee_interfaces/msg/detail/pickee_mobile_arrival__struct.h"
#include "shopee_interfaces/msg/detail/pickee_mobile_arrival__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

bool shopee_interfaces__msg__pose2_d__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * shopee_interfaces__msg__pose2_d__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool shopee_interfaces__msg__pickee_mobile_arrival__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[65];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("shopee_interfaces.msg._pickee_mobile_arrival.PickeeMobileArrival", full_classname_dest, 64) == 0);
  }
  shopee_interfaces__msg__PickeeMobileArrival * ros_message = _ros_message;
  {  // robot_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "robot_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->robot_id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // order_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "order_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->order_id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // location_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "location_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->location_id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // final_pose
    PyObject * field = PyObject_GetAttrString(_pymsg, "final_pose");
    if (!field) {
      return false;
    }
    if (!shopee_interfaces__msg__pose2_d__convert_from_py(field, &ros_message->final_pose)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // position_error
    PyObject * field = PyObject_GetAttrString(_pymsg, "position_error");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->position_error = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // travel_time
    PyObject * field = PyObject_GetAttrString(_pymsg, "travel_time");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->travel_time = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // message
    PyObject * field = PyObject_GetAttrString(_pymsg, "message");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->message, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * shopee_interfaces__msg__pickee_mobile_arrival__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PickeeMobileArrival */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("shopee_interfaces.msg._pickee_mobile_arrival");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PickeeMobileArrival");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  shopee_interfaces__msg__PickeeMobileArrival * ros_message = (shopee_interfaces__msg__PickeeMobileArrival *)raw_ros_message;
  {  // robot_id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->robot_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "robot_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // order_id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->order_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "order_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // location_id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->location_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "location_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // final_pose
    PyObject * field = NULL;
    field = shopee_interfaces__msg__pose2_d__convert_to_py(&ros_message->final_pose);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "final_pose", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // position_error
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->position_error);
    {
      int rc = PyObject_SetAttrString(_pymessage, "position_error", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // travel_time
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->travel_time);
    {
      int rc = PyObject_SetAttrString(_pymessage, "travel_time", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // message
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->message.data,
      strlen(ros_message->message.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "message", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
