cmake_minimum_required(VERSION 3.10)
project(packee_arm)

# ament 패키지 설정
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(shopee_interfaces REQUIRED)

# C++ 실행 파일 정의
add_executable(packee_arm_controller
  src/packee_arm_controller.cpp
  src/visual_servo_module.cpp
  src/arm_driver_proxy.cpp
  src/gripper_controller.cpp
  src/execution_manager.cpp
)
ament_target_dependencies(packee_arm_controller rclcpp rosidl_runtime_cpp shopee_interfaces)
target_include_directories(packee_arm_controller PUBLIC
  include
  ${rosidl_runtime_cpp_INCLUDE_DIRS}
)
target_compile_features(packee_arm_controller PUBLIC cxx_std_17)

install(
  TARGETS packee_arm_controller
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(mock_packee_main src/mock_packee_main.cpp)
ament_target_dependencies(mock_packee_main rclcpp rosidl_runtime_cpp shopee_interfaces)
target_include_directories(mock_packee_main PUBLIC
  ${rosidl_runtime_cpp_INCLUDE_DIRS}
)
target_compile_features(mock_packee_main PUBLIC cxx_std_17)

install(
  TARGETS mock_packee_main
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
