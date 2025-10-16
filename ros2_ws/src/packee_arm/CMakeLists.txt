cmake_minimum_required(VERSION 3.10)
project(packee_arm)

# ament 패키지 설정
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shopee_interfaces REQUIRED)

# Python 패키지 설치 (기존 rclpy 노드 유지)
ament_python_install_package(${PROJECT_NAME})

# Python 실행 스크립트 설치
install(
  PROGRAMS
    scripts/arm_controller.py
    scripts/mock_packee_main.py
  DESTINATION lib/${PROJECT_NAME}
)

# C++ 실행 파일 정의
add_executable(packee_arm_controller src/packee_arm_controller.cpp)
ament_target_dependencies(packee_arm_controller rclcpp shopee_interfaces)
target_compile_features(packee_arm_controller PUBLIC cxx_std_17)

install(
  TARGETS packee_arm_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
