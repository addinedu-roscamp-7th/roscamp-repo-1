cmake_minimum_required(VERSION 3.8)
project(pickee_mobile_wonho)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# Navigation2 packages
find_package(nav2_mppi_controller REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav_msgs REQUIRED)

# Geometry and Transform packages
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Shopee Interface package
find_package(shopee_interfaces REQUIRED)

# Standard message packages
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Additional utilities
find_package(pluginlib REQUIRED)

# Include directories
include_directories(
  include
)

# Custom Planner 라이브러리 생성
set(library_name ${PROJECT_NAME}__custom_planner)

add_library(${library_name} SHARED
  src/custom_planner.cpp
)

ament_target_dependencies(${library_name}
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  geometry_msgs
  tf2_ros
  pluginlib
)

# Custom Planner Simulation 라이브러리 생성
set(library_name_sim ${PROJECT_NAME}__custom_planner_simulation)

add_library(${library_name_sim} SHARED
  src/custom_planner_simulation.cpp
)

ament_target_dependencies(${library_name_sim}
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  geometry_msgs
  tf2_ros
  pluginlib
)

# Install custom_planner_simulation library
install(TARGETS ${library_name_sim}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Export custom_planner_simulation library
ament_export_libraries(${library_name_sim})

# Create executable for pickee mobile node (Shopee Interface 통신)
add_executable(pickee_mobile_wonho_node
  src/pickee_mobile_wonho_node.cpp
)

# Dependencies for pickee_mobile_wonho_node (Shopee Interface 포함)
ament_target_dependencies(pickee_mobile_wonho_node
  rclcpp
  rclcpp_action
  geometry_msgs
  nav_msgs
  nav2_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  std_msgs
  std_srvs
  shopee_interfaces
)

# Hardware bringup executable removed

# Install executables
install(TARGETS
  pickee_mobile_wonho_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files (if exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

# Install maps (if exists)  
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/map")
  install(DIRECTORY
    map
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

# Install params (if exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/params")
  install(DIRECTORY
    params
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

# Install rviz configs (if exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rviz")
  install(DIRECTORY
    rviz
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

# Install urdf files (if exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/urdf")
  install(DIRECTORY
    urdf
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

# Install worlds (if exists)
install(DIRECTORY
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

# Install worlds (if exists)
install(DIRECTORY
  models/
  DESTINATION share/${PROJECT_NAME}/models
)

# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/primitives")
#   install(
#     DIRECTORY primitives
#     DESTINATION share/${PROJECT_NAME}
#   )
# endif()

# Install custom_planner library
install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Export custom_planner library
ament_export_libraries(${library_name})

# Install pluginlib description
pluginlib_export_plugin_description_file(nav2_core pluginlib.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  
  # the following line skips the linter which checks for copyrights
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  set(ament_cmake_cpplint_FOUND TRUE)
  
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
