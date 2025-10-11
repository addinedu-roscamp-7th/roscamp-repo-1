// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from shopee_interfaces:msg/PickeeDetectedProduct.idl
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
#include "shopee_interfaces/msg/detail/pickee_detected_product__struct.h"
#include "shopee_interfaces/msg/detail/pickee_detected_product__functions.h"

bool shopee_interfaces__msg__b_box__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * shopee_interfaces__msg__b_box__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool shopee_interfaces__msg__pickee_detected_product__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[69];
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
    assert(strncmp("shopee_interfaces.msg._pickee_detected_product.PickeeDetectedProduct", full_classname_dest, 68) == 0);
  }
  shopee_interfaces__msg__PickeeDetectedProduct * ros_message = _ros_message;
  {  // product_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "product_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->product_id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // bbox_number
    PyObject * field = PyObject_GetAttrString(_pymsg, "bbox_number");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->bbox_number = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // bbox_coords
    PyObject * field = PyObject_GetAttrString(_pymsg, "bbox_coords");
    if (!field) {
      return false;
    }
    if (!shopee_interfaces__msg__b_box__convert_from_py(field, &ros_message->bbox_coords)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // confidence
    PyObject * field = PyObject_GetAttrString(_pymsg, "confidence");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->confidence = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * shopee_interfaces__msg__pickee_detected_product__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PickeeDetectedProduct */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("shopee_interfaces.msg._pickee_detected_product");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PickeeDetectedProduct");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  shopee_interfaces__msg__PickeeDetectedProduct * ros_message = (shopee_interfaces__msg__PickeeDetectedProduct *)raw_ros_message;
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
  {  // bbox_number
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->bbox_number);
    {
      int rc = PyObject_SetAttrString(_pymessage, "bbox_number", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // bbox_coords
    PyObject * field = NULL;
    field = shopee_interfaces__msg__b_box__convert_to_py(&ros_message->bbox_coords);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "bbox_coords", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // confidence
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->confidence);
    {
      int rc = PyObject_SetAttrString(_pymessage, "confidence", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
