// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from shopee_interfaces:msg/ProductLocation.idl
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
#include "shopee_interfaces/msg/detail/product_location__struct.h"
#include "shopee_interfaces/msg/detail/product_location__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool shopee_interfaces__msg__product_location__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[56];
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
    assert(strncmp("shopee_interfaces.msg._product_location.ProductLocation", full_classname_dest, 55) == 0);
  }
  shopee_interfaces__msg__ProductLocation * ros_message = _ros_message;
  {  // product_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "product_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->product_id = (int32_t)PyLong_AsLong(field);
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
  {  // section_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "section_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->section_id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // quantity
    PyObject * field = PyObject_GetAttrString(_pymsg, "quantity");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->quantity = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * shopee_interfaces__msg__product_location__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of ProductLocation */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("shopee_interfaces.msg._product_location");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "ProductLocation");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  shopee_interfaces__msg__ProductLocation * ros_message = (shopee_interfaces__msg__ProductLocation *)raw_ros_message;
  {  // product_id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->product_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "product_id", field);
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
  {  // section_id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->section_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "section_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // quantity
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->quantity);
    {
      int rc = PyObject_SetAttrString(_pymessage, "quantity", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
