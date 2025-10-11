// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from shopee_interfaces:msg/PickeeProductDetection.idl
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
#include "shopee_interfaces/msg/detail/pickee_product_detection__struct.h"
#include "shopee_interfaces/msg/detail/pickee_product_detection__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "shopee_interfaces/msg/detail/pickee_detected_product__functions.h"
// end nested array functions include
bool shopee_interfaces__msg__pickee_detected_product__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * shopee_interfaces__msg__pickee_detected_product__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool shopee_interfaces__msg__pickee_product_detection__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[71];
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
    assert(strncmp("shopee_interfaces.msg._pickee_product_detection.PickeeProductDetection", full_classname_dest, 70) == 0);
  }
  shopee_interfaces__msg__PickeeProductDetection * ros_message = _ros_message;
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
  {  // products
    PyObject * field = PyObject_GetAttrString(_pymsg, "products");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'products'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!shopee_interfaces__msg__PickeeDetectedProduct__Sequence__init(&(ros_message->products), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create shopee_interfaces__msg__PickeeDetectedProduct__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    shopee_interfaces__msg__PickeeDetectedProduct * dest = ros_message->products.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!shopee_interfaces__msg__pickee_detected_product__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * shopee_interfaces__msg__pickee_product_detection__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PickeeProductDetection */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("shopee_interfaces.msg._pickee_product_detection");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PickeeProductDetection");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  shopee_interfaces__msg__PickeeProductDetection * ros_message = (shopee_interfaces__msg__PickeeProductDetection *)raw_ros_message;
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
  {  // products
    PyObject * field = NULL;
    size_t size = ros_message->products.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    shopee_interfaces__msg__PickeeDetectedProduct * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->products.data[i]);
      PyObject * pyitem = shopee_interfaces__msg__pickee_detected_product__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "products", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
