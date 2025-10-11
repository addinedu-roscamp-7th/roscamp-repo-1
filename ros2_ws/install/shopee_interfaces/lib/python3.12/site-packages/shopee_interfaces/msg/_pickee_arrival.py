# generated from rosidl_generator_py/resource/_idl.py.em
# with input from shopee_interfaces:msg/PickeeArrival.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PickeeArrival(type):
    """Metaclass of message 'PickeeArrival'."""

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
                'shopee_interfaces.msg.PickeeArrival')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__pickee_arrival
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__pickee_arrival
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__pickee_arrival
            cls._TYPE_SUPPORT = module.type_support_msg__msg__pickee_arrival
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__pickee_arrival

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PickeeArrival(metaclass=Metaclass_PickeeArrival):
    """Message class 'PickeeArrival'."""

    __slots__ = [
        '_robot_id',
        '_order_id',
        '_location_id',
        '_section_id',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'robot_id': 'int32',
        'order_id': 'int32',
        'location_id': 'int32',
        'section_id': 'int32',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
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
        self.robot_id = kwargs.get('robot_id', int())
        self.order_id = kwargs.get('order_id', int())
        self.location_id = kwargs.get('location_id', int())
        self.section_id = kwargs.get('section_id', int())

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
        if self.robot_id != other.robot_id:
            return False
        if self.order_id != other.order_id:
            return False
        if self.location_id != other.location_id:
            return False
        if self.section_id != other.section_id:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def robot_id(self):
        """Message field 'robot_id'."""
        return self._robot_id

    @robot_id.setter
    def robot_id(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'robot_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'robot_id' field must be an integer in [-2147483648, 2147483647]"
        self._robot_id = value

    @builtins.property
    def order_id(self):
        """Message field 'order_id'."""
        return self._order_id

    @order_id.setter
    def order_id(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'order_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'order_id' field must be an integer in [-2147483648, 2147483647]"
        self._order_id = value

    @builtins.property
    def location_id(self):
        """Message field 'location_id'."""
        return self._location_id

    @location_id.setter
    def location_id(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'location_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'location_id' field must be an integer in [-2147483648, 2147483647]"
        self._location_id = value

    @builtins.property
    def section_id(self):
        """Message field 'section_id'."""
        return self._section_id

    @section_id.setter
    def section_id(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'section_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'section_id' field must be an integer in [-2147483648, 2147483647]"
        self._section_id = value
