# generated from rosidl_generator_py/resource/_idl.py.em
# with input from shopee_interfaces:srv/PackeeVisionVerifyPackingComplete.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PackeeVisionVerifyPackingComplete_Request(type):
    """Metaclass of message 'PackeeVisionVerifyPackingComplete_Request'."""

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
                'shopee_interfaces.srv.PackeeVisionVerifyPackingComplete_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__packee_vision_verify_packing_complete__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__packee_vision_verify_packing_complete__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__packee_vision_verify_packing_complete__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__packee_vision_verify_packing_complete__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__packee_vision_verify_packing_complete__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PackeeVisionVerifyPackingComplete_Request(metaclass=Metaclass_PackeeVisionVerifyPackingComplete_Request):
    """Message class 'PackeeVisionVerifyPackingComplete_Request'."""

    __slots__ = [
        '_robot_id',
        '_order_id',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'robot_id': 'int32',
        'order_id': 'int32',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
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


# Import statements for member types

# Member 'remaining_product_ids'
import array  # noqa: E402, I100

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_PackeeVisionVerifyPackingComplete_Response(type):
    """Metaclass of message 'PackeeVisionVerifyPackingComplete_Response'."""

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
                'shopee_interfaces.srv.PackeeVisionVerifyPackingComplete_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__packee_vision_verify_packing_complete__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__packee_vision_verify_packing_complete__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__packee_vision_verify_packing_complete__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__packee_vision_verify_packing_complete__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__packee_vision_verify_packing_complete__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PackeeVisionVerifyPackingComplete_Response(metaclass=Metaclass_PackeeVisionVerifyPackingComplete_Response):
    """Message class 'PackeeVisionVerifyPackingComplete_Response'."""

    __slots__ = [
        '_cart_empty',
        '_remaining_items',
        '_remaining_product_ids',
        '_message',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'cart_empty': 'boolean',
        'remaining_items': 'int32',
        'remaining_product_ids': 'sequence<int32>',
        'message': 'string',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int32')),  # noqa: E501
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
        self.cart_empty = kwargs.get('cart_empty', bool())
        self.remaining_items = kwargs.get('remaining_items', int())
        self.remaining_product_ids = array.array('i', kwargs.get('remaining_product_ids', []))
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
        if self.cart_empty != other.cart_empty:
            return False
        if self.remaining_items != other.remaining_items:
            return False
        if self.remaining_product_ids != other.remaining_product_ids:
            return False
        if self.message != other.message:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def cart_empty(self):
        """Message field 'cart_empty'."""
        return self._cart_empty

    @cart_empty.setter
    def cart_empty(self, value):
        if self._check_fields:
            assert \
                isinstance(value, bool), \
                "The 'cart_empty' field must be of type 'bool'"
        self._cart_empty = value

    @builtins.property
    def remaining_items(self):
        """Message field 'remaining_items'."""
        return self._remaining_items

    @remaining_items.setter
    def remaining_items(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'remaining_items' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'remaining_items' field must be an integer in [-2147483648, 2147483647]"
        self._remaining_items = value

    @builtins.property
    def remaining_product_ids(self):
        """Message field 'remaining_product_ids'."""
        return self._remaining_product_ids

    @remaining_product_ids.setter
    def remaining_product_ids(self, value):
        if self._check_fields:
            if isinstance(value, array.array):
                assert value.typecode == 'i', \
                    "The 'remaining_product_ids' array.array() must have the type code of 'i'"
                self._remaining_product_ids = value
                return
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'remaining_product_ids' field must be a set or sequence and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._remaining_product_ids = array.array('i', value)

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


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_PackeeVisionVerifyPackingComplete_Event(type):
    """Metaclass of message 'PackeeVisionVerifyPackingComplete_Event'."""

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
                'shopee_interfaces.srv.PackeeVisionVerifyPackingComplete_Event')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__packee_vision_verify_packing_complete__event
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__packee_vision_verify_packing_complete__event
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__packee_vision_verify_packing_complete__event
            cls._TYPE_SUPPORT = module.type_support_msg__srv__packee_vision_verify_packing_complete__event
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__packee_vision_verify_packing_complete__event

            from service_msgs.msg import ServiceEventInfo
            if ServiceEventInfo.__class__._TYPE_SUPPORT is None:
                ServiceEventInfo.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PackeeVisionVerifyPackingComplete_Event(metaclass=Metaclass_PackeeVisionVerifyPackingComplete_Event):
    """Message class 'PackeeVisionVerifyPackingComplete_Event'."""

    __slots__ = [
        '_info',
        '_request',
        '_response',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'info': 'service_msgs/ServiceEventInfo',
        'request': 'sequence<shopee_interfaces/PackeeVisionVerifyPackingComplete_Request, 1>',
        'response': 'sequence<shopee_interfaces/PackeeVisionVerifyPackingComplete_Response, 1>',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['service_msgs', 'msg'], 'ServiceEventInfo'),  # noqa: E501
        rosidl_parser.definition.BoundedSequence(rosidl_parser.definition.NamespacedType(['shopee_interfaces', 'srv'], 'PackeeVisionVerifyPackingComplete_Request'), 1),  # noqa: E501
        rosidl_parser.definition.BoundedSequence(rosidl_parser.definition.NamespacedType(['shopee_interfaces', 'srv'], 'PackeeVisionVerifyPackingComplete_Response'), 1),  # noqa: E501
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
        from service_msgs.msg import ServiceEventInfo
        self.info = kwargs.get('info', ServiceEventInfo())
        self.request = kwargs.get('request', [])
        self.response = kwargs.get('response', [])

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
        if self.info != other.info:
            return False
        if self.request != other.request:
            return False
        if self.response != other.response:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def info(self):
        """Message field 'info'."""
        return self._info

    @info.setter
    def info(self, value):
        if self._check_fields:
            from service_msgs.msg import ServiceEventInfo
            assert \
                isinstance(value, ServiceEventInfo), \
                "The 'info' field must be a sub message of type 'ServiceEventInfo'"
        self._info = value

    @builtins.property
    def request(self):
        """Message field 'request'."""
        return self._request

    @request.setter
    def request(self, value):
        if self._check_fields:
            from shopee_interfaces.srv import PackeeVisionVerifyPackingComplete_Request
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) <= 1 and
                 all(isinstance(v, PackeeVisionVerifyPackingComplete_Request) for v in value) and
                 True), \
                "The 'request' field must be a set or sequence with length <= 1 and each value of type 'PackeeVisionVerifyPackingComplete_Request'"
        self._request = value

    @builtins.property
    def response(self):
        """Message field 'response'."""
        return self._response

    @response.setter
    def response(self, value):
        if self._check_fields:
            from shopee_interfaces.srv import PackeeVisionVerifyPackingComplete_Response
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) <= 1 and
                 all(isinstance(v, PackeeVisionVerifyPackingComplete_Response) for v in value) and
                 True), \
                "The 'response' field must be a set or sequence with length <= 1 and each value of type 'PackeeVisionVerifyPackingComplete_Response'"
        self._response = value


class Metaclass_PackeeVisionVerifyPackingComplete(type):
    """Metaclass of service 'PackeeVisionVerifyPackingComplete'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('shopee_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'shopee_interfaces.srv.PackeeVisionVerifyPackingComplete')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__packee_vision_verify_packing_complete

            from shopee_interfaces.srv import _packee_vision_verify_packing_complete
            if _packee_vision_verify_packing_complete.Metaclass_PackeeVisionVerifyPackingComplete_Request._TYPE_SUPPORT is None:
                _packee_vision_verify_packing_complete.Metaclass_PackeeVisionVerifyPackingComplete_Request.__import_type_support__()
            if _packee_vision_verify_packing_complete.Metaclass_PackeeVisionVerifyPackingComplete_Response._TYPE_SUPPORT is None:
                _packee_vision_verify_packing_complete.Metaclass_PackeeVisionVerifyPackingComplete_Response.__import_type_support__()
            if _packee_vision_verify_packing_complete.Metaclass_PackeeVisionVerifyPackingComplete_Event._TYPE_SUPPORT is None:
                _packee_vision_verify_packing_complete.Metaclass_PackeeVisionVerifyPackingComplete_Event.__import_type_support__()


class PackeeVisionVerifyPackingComplete(metaclass=Metaclass_PackeeVisionVerifyPackingComplete):
    from shopee_interfaces.srv._packee_vision_verify_packing_complete import PackeeVisionVerifyPackingComplete_Request as Request
    from shopee_interfaces.srv._packee_vision_verify_packing_complete import PackeeVisionVerifyPackingComplete_Response as Response
    from shopee_interfaces.srv._packee_vision_verify_packing_complete import PackeeVisionVerifyPackingComplete_Event as Event

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
