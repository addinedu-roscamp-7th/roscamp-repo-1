cmake_minimum_required(VERSION 3.8)
project(vicpinky_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

#------------------------------------------------------------------------------------

# target_include_directories(my_first_node_cpp PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)

# bringup
add_executable(bringup src/bringup.cpp)
target_link_libraries(bringup 
  Boost::system 
  rclcpp::rclcpp
  tf2_ros::tf2_ros
)
ament_target_dependencies(bringup
  sensor_msgs
  nav_msgs
)
target_include_directories(bringup PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}  
)
target_compile_features(bringup PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS bringup
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(
    DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
)
#------------------------------------------------------------------------------------

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
