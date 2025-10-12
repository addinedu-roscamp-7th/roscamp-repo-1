#!/usr/bin/env python3
"""
Main Service 직접 실행 스크립트
"""
import sys
import os

# ROS2 패키지 경로 추가
sys.path.insert(0, os.path.expanduser('~/dev_ws/Shopee/ros2_ws/install/shopee_main_service/lib/python3.12/site-packages'))
sys.path.insert(0, os.path.expanduser('~/dev_ws/Shopee/ros2_ws/install/shopee_interfaces/lib/python3.12/site-packages'))

# Main Service 실행
from shopee_main_service.main_service_node import main

if __name__ == "__main__":
    main()
