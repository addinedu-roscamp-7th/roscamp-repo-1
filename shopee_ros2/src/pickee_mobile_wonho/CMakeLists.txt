cmake_minimum_required(VERSION 3.8)
project(pickee_mobile_wonho)

# C++ 표준 설정
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++17 표준 설정
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 컴파일러 최적화 옵션
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -flto)
endif()

# ROS2 의존성 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(shopee_interfaces REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_util REQUIRED)

# Eigen3 찾기
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# 의존성 목록
set(dependencies
  rclcpp
  rclcpp_components
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  shopee_interfaces
  nav2_common
  nav2_core
  nav2_msgs
  nav2_util
  Eigen3
)

# 헤더 파일 디렉터리
include_directories(include)

# 라이브러리 생성 - 상태 기계
add_library(${PROJECT_NAME}_state_machine
  src/state_machine.cpp
  src/states/idle_state.cpp
  src/states/moving_state.cpp
  src/states/stopped_state.cpp
  src/states/charging_state.cpp
  src/states/error_state.cpp
)

target_include_directories(${PROJECT_NAME}_state_machine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_state_machine ${dependencies})

# 라이브러리 생성 - 컴포넌트들
add_library(${PROJECT_NAME}_components
  src/components/localization_component.cpp
  src/components/path_planning_component.cpp
  src/components/motion_control_component.cpp
  src/components/sensor_manager.cpp
  src/components/battery_manager.cpp
  src/components/arrival_detector.cpp
)

target_include_directories(${PROJECT_NAME}_components PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_components ${dependencies})

# 라이브러리 생성 - 통신 인터페이스
add_library(${PROJECT_NAME}_communication
  src/communication_interface.cpp
)

target_include_directories(${PROJECT_NAME}_communication PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_communication ${dependencies})

# 메인 실행 파일 생성
add_executable(${PROJECT_NAME}_controller
  src/main.cpp
  src/mobile_controller.cpp
)

target_include_directories(${PROJECT_NAME}_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 라이브러리 링크
target_link_libraries(${PROJECT_NAME}_controller
  ${PROJECT_NAME}_state_machine
  ${PROJECT_NAME}_components
  ${PROJECT_NAME}_communication
)

ament_target_dependencies(${PROJECT_NAME}_controller ${dependencies})

# 설치
install(TARGETS
  ${PROJECT_NAME}_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  ${PROJECT_NAME}_state_machine
  ${PROJECT_NAME}_components
  ${PROJECT_NAME}_communication
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY include/
  DESTINATION include/
)

# 테스트
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  find_package(ament_cmake_gtest REQUIRED)
  
  # 상태 기계 테스트
  ament_add_gtest(test_state_machine
    test/test_state_machine.cpp
  )
  
  target_include_directories(test_state_machine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  
  target_link_libraries(test_state_machine
    ${PROJECT_NAME}_state_machine
  )
  
  ament_target_dependencies(test_state_machine ${dependencies})
  
  # 개별 컴포넌트 테스트
  ament_add_gtest(test_localization_component
    test/test_localization_component.cpp
  )
  
  target_include_directories(test_localization_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  
  target_link_libraries(test_localization_component
    ${PROJECT_NAME}_components
  )
  
  ament_target_dependencies(test_localization_component ${dependencies})
  
  ament_add_gtest(test_path_planning_component
    test/test_path_planning_component.cpp
  )
  
  target_include_directories(test_path_planning_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  
  target_link_libraries(test_path_planning_component
    ${PROJECT_NAME}_components
  )
  
  ament_target_dependencies(test_path_planning_component ${dependencies})
  
  ament_add_gtest(test_motion_control_component
    test/test_motion_control_component.cpp
  )
  
  target_include_directories(test_motion_control_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  
  target_link_libraries(test_motion_control_component
    ${PROJECT_NAME}_components
  )
  
  ament_target_dependencies(test_motion_control_component ${dependencies})
endif()

ament_export_include_directories(include)
ament_export_libraries(
  ${PROJECT_NAME}_state_machine
  ${PROJECT_NAME}_components  
  ${PROJECT_NAME}_communication
)
ament_export_dependencies(${dependencies})

ament_package()