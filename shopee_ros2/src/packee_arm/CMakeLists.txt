cmake_minimum_required(VERSION 3.10)
project(packee_arm)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(shopee_interfaces REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
  include
  ${rosidl_runtime_cpp_INCLUDE_DIRS}
)

set(COMMON_DEPENDENCIES
  rclcpp
  rosidl_runtime_cpp
  shopee_interfaces
  trajectory_msgs
  sensor_msgs
  geometry_msgs
)

# -----------------------------------------------------------
# 🧱 1️⃣ 라이브러리 모듈로 묶기
# -----------------------------------------------------------
add_library(packee_arm_core
  src/arm_driver_proxy.cpp
  src/execution_manager.cpp
  src/gripper_controller.cpp
  src/visual_servo_module.cpp
)
ament_target_dependencies(packee_arm_core ${COMMON_DEPENDENCIES})
target_include_directories(packee_arm_core PUBLIC include)
target_compile_features(packee_arm_core PUBLIC cxx_std_17)

# -----------------------------------------------------------
# ⚙️ 2️⃣ 실행 파일들
# -----------------------------------------------------------

# myCobot 제어 노드
add_executable(simple_mycobot_motion src/simple_mycobot_motion.cpp)
ament_target_dependencies(simple_mycobot_motion ${COMMON_DEPENDENCIES})
target_link_libraries(simple_mycobot_motion packee_arm_core)

# 실제 컨트롤러 노드
add_executable(packee_arm_controller src/packee_arm_controller.cpp)
ament_target_dependencies(packee_arm_controller ${COMMON_DEPENDENCIES})
target_link_libraries(packee_arm_controller packee_arm_core)

# 모의 통신 테스트 노드
add_executable(mock_packee_main src/mock_packee_main.cpp)
ament_target_dependencies(mock_packee_main ${COMMON_DEPENDENCIES})
target_link_libraries(mock_packee_main packee_arm_core)

# -----------------------------------------------------------
# 📦 설치
# -----------------------------------------------------------
install(
  TARGETS
    simple_mycobot_motion
    packee_arm_controller
    mock_packee_main
    packee_arm_core
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY resource DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/jetcobot_bridge.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
