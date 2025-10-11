# generated from rosidl_generator_py/resource/_idl.py.em
# with input from shopee_interfaces:msg/PackeeDetectedProduct.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PackeeDetectedProduct(type):
    """Metaclass of message 'PackeeDetectedProduct'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('shopee_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'shopee_interfaces.msg.PackeeDetectedProduct')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__packee_detected_product
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__packee_detected_product
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__packee_detected_product
            cls._TYPE_SUPPORT = module.type_support_msg__msg__packee_detected_product
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__packee_detected_product

            from shopee_interfaces.msg import BBox
            if BBox.__class__._TYPE_SUPPORT is None:
                BBox.__class__.__import_type_support__()

            from shopee_interfaces.msg import Point3D
            if Point3D.__class__._TYPE_SUPPORT is None:
                Point3D.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PackeeDetectedProduct(metaclass=Metaclass_PackeeDetectedProduct):
    """Message class 'PackeeDetectedProduct'."""

    __slots__ = [
        '_product_id',
        '_bbox',
        '_confidence',
        '_position',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'product_id': 'int32',
        'bbox': 'shopee_interfaces/BBox',
        'confidence': 'float',
        'position': 'shopee_interfaces/Point3D',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['shopee_interfaces', 'msg'], 'BBox'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['shopee_interfaces', 'msg'], 'Point3D'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        if 'check_fields' in kwargs:
            self._check_fields = kwargs['check_fields']
        else:
            self._check_fields = ros_python_check_fields == '1'
        if self._check_fields:
            assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
                'Invalid arguments passed to constructor: %s' % \
                ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.product_id = kwargs.get('product_id', int())
        from shopee_interfaces.msg import BBox
        self.bbox = kwargs.get('bbox', BBox())
        self.confidence = kwargs.get('confidence', float())
        from shopee_interfaces.msg import Point3D
        self.position = kwargs.get('position', Point3D())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.get_fields_and_field_types().keys(), self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    if self._check_fields:
                        assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.product_id != other.product_id:
            return False
        if self.bbox != other.bbox:
            return False
        if self.confidence != other.confidence:
            return False
        if self.position != other.position:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def product_id(self):
        """Message field 'product_id'."""
        return self._product_id

    @product_id.setter
    def product_id(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'product_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'product_id' field must be an integer in [-2147483648, 2147483647]"
        self._product_id = value

    @builtins.property
    def bbox(self):
        """Message field 'bbox'."""
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        if self._check_fields:
            from shopee_interfaces.msg import BBox
            assert \
                isinstance(value, BBox), \
                "The 'bbox' field must be a sub message of type 'BBox'"
        self._bbox = value

    @builtins.property
    def confidence(self):
        """Message field 'confidence'."""
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'confidence' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'confidence' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._confidence = value

    @builtins.property
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if self._check_fields:
            from shopee_interfaces.msg import Point3D
            assert \
                isinstance(value, Point3D), \
                "The 'position' field must be a sub message of type 'Point3D'"
        self._position = value
