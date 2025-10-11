import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jinhyuk2me/dev_ws/Shopee/ros2_ws/install/shopee_main_service'
