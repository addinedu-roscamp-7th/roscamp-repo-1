#!/bin/bash
# Main Service 직접 실행 스크립트

# ROS2 환경 source
source ~/dev_ws/Shopee/ros2_ws/install/setup.bash

# Python으로 직접 실행
python3 -m shopee_main_service.main_service_node
