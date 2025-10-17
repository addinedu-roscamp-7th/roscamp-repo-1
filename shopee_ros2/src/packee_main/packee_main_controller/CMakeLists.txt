cmake_minimum_required(VERSION 3.8)
project(packee_main_controller)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(shopee_interfaces REQUIRED)

add_executable(packee_main_service src/packee_main_service.cpp)
ament_target_dependencies(packee_main_service rclcpp rosidl_runtime_cpp shopee_interfaces)
target_include_directories(packee_main_service PUBLIC
  ${rosidl_runtime_cpp_INCLUDE_DIRS}
)
target_compile_features(packee_main_service PUBLIC cxx_std_17)

install(
  TARGETS packee_main_service
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(packee_main_topic src/packee_main_topic.cpp)
ament_target_dependencies(packee_main_topic rclcpp rosidl_runtime_cpp shopee_interfaces)
target_include_directories(packee_main_topic PUBLIC
  ${rosidl_runtime_cpp_INCLUDE_DIRS}
)
target_compile_features(packee_main_topic PUBLIC cxx_std_17)

install(
  TARGETS packee_main_topic
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