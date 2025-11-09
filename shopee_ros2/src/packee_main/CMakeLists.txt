cmake_minimum_required(VERSION 3.8)
project(packee_main)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(shopee_interfaces REQUIRED)

add_executable(packee_state_manager src/packee_state_manager.cpp)
ament_target_dependencies(packee_state_manager rclcpp rosidl_runtime_cpp shopee_interfaces)
target_include_directories(packee_state_manager PUBLIC
  ${rosidl_runtime_cpp_INCLUDE_DIRS}
)
target_compile_features(packee_state_manager PUBLIC cxx_std_17)

install(
  TARGETS packee_state_manager
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(packee_check_availability src/packee_check_availability.cpp)
ament_target_dependencies(packee_check_availability rclcpp rosidl_runtime_cpp shopee_interfaces)
target_include_directories(packee_check_availability PUBLIC
  ${rosidl_runtime_cpp_INCLUDE_DIRS}
)
target_compile_features(packee_check_availability PUBLIC cxx_std_17)

install(
  TARGETS packee_check_availability
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(packee_packing src/packee_packing.cpp)
ament_target_dependencies(packee_packing rclcpp rosidl_runtime_cpp shopee_interfaces)
target_include_directories(packee_packing PUBLIC
  ${rosidl_runtime_cpp_INCLUDE_DIRS}
)
target_compile_features(packee_packing PUBLIC cxx_std_17)

install(
  TARGETS packee_packing
  DESTINATION lib/${PROJECT_NAME}
)

# Launch 파일 설치
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()