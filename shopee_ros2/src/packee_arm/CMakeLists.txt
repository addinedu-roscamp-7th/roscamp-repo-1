cmake_minimum_required(VERSION 3.10)
project(packee_arm)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY resource DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/jetcobot_bridge.py
  src/pymycobot_left
  src/pymycobot_dual
  src/pymycobot_right
  DESTINATION lib/${PROJECT_NAME})

ament_package()
