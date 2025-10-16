cmake_minimum_required(VERSION 3.10)
project(packee_arm)

# ament 패키지 설정
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shopee_interfaces REQUIRED)

# C++ 실행 파일 정의
add_executable(packee_arm_controller src/packee_arm_controller.cpp)
ament_target_dependencies(packee_arm_controller rclcpp shopee_interfaces)
target_compile_features(packee_arm_controller PUBLIC cxx_std_17)

install(
  TARGETS packee_arm_controller
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(mock_packee_main src/mock_packee_main.cpp)
ament_target_dependencies(mock_packee_main rclcpp shopee_interfaces)
target_compile_features(mock_packee_main PUBLIC cxx_std_17)

install(
  TARGETS mock_packee_main
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
