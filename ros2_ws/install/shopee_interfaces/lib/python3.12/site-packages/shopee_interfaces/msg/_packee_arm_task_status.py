# generated from rosidl_generator_py/resource/_idl.py.em
# with input from shopee_interfaces:msg/PackeeArmTaskStatus.idl
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


class Metaclass_PackeeArmTaskStatus(type):
    """Metaclass of message 'PackeeArmTaskStatus'."""

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
                'shopee_interfaces.msg.PackeeArmTaskStatus')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__packee_arm_task_status
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__packee_arm_task_status
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__packee_arm_task_status
            cls._TYPE_SUPPORT = module.type_support_msg__msg__packee_arm_task_status
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__packee_arm_task_status

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PackeeArmTaskStatus(metaclass=Metaclass_PackeeArmTaskStatus):
    """Message class 'PackeeArmTaskStatus'."""

    __slots__ = [
        '_robot_id',
        '_order_id',
        '_product_id',
        '_arm_side',
        '_status',
        '_current_phase',
        '_progress',
        '_message',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'robot_id': 'int32',
        'order_id': 'int32',
        'product_id': 'int32',
        'arm_side': 'string',
        'status': 'string',
        'current_phase': 'string',
        'progress': 'float',
        'message': 'string',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
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
        self.product_id = kwargs.get('product_id', int())
        self.arm_side = kwargs.get('arm_side', str())
        self.status = kwargs.get('status', str())
        self.current_phase = kwargs.get('current_phase', str())
        self.progress = kwargs.get('progress', float())
        self.message = kwargs.get('message', str())

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
        if self.product_id != other.product_id:
            return False
        if self.arm_side != other.arm_side:
            return False
        if self.status != other.status:
            return False
        if self.current_phase != other.current_phase:
            return False
        if self.progress != other.progress:
            return False
        if self.message != other.message:
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
    def arm_side(self):
        """Message field 'arm_side'."""
        return self._arm_side

    @arm_side.setter
    def arm_side(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'arm_side' field must be of type 'str'"
        self._arm_side = value

    @builtins.property
    def status(self):
        """Message field 'status'."""
        return self._status

    @status.setter
    def status(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'status' field must be of type 'str'"
        self._status = value

    @builtins.property
    def current_phase(self):
        """Message field 'current_phase'."""
        return self._current_phase

    @current_phase.setter
    def current_phase(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'current_phase' field must be of type 'str'"
        self._current_phase = value

    @builtins.property
    def progress(self):
        """Message field 'progress'."""
        return self._progress

    @progress.setter
    def progress(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'progress' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'progress' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._progress = value

    @builtins.property
    def message(self):
        """Message field 'message'."""
        return self._message

    @message.setter
    def message(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'message' field must be of type 'str'"
        self._message = value
