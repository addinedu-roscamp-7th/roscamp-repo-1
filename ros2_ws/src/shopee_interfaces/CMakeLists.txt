cmake_minimum_required(VERSION 3.8)
project(shopee_interfaces)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msg_files
  "msg/ArmPoseStatus.msg"
  "msg/BBox.msg"
  "msg/Obstacle.msg"
  "msg/PackeeArmTaskStatus.msg"
  "msg/PackeeAvailability.msg"
  "msg/PackeeDetectedProduct.msg"
  "msg/PackeePackingComplete.msg"
  "msg/PackeeRobotStatus.msg"
  "msg/PickeeArmTaskStatus.msg"
  "msg/PickeeArrival.msg"
  "msg/PickeeCartHandover.msg"
  "msg/PickeeDetectedProduct.msg"
  "msg/PickeeMobileArrival.msg"
  "msg/PickeeMobilePose.msg"
  "msg/PickeeMobileSpeedControl.msg"
  "msg/PickeeMoveStatus.msg"
  "msg/PickeeProductDetection.msg"
  "msg/PickeeProductSelection.msg"
  "msg/PickeeRobotStatus.msg"
  "msg/PickeeVisionCartCheck.msg"
  "msg/PickeeVisionDetection.msg"
  "msg/PickeeVisionObstacles.msg"
  "msg/PickeeVisionStaffRegister.msg"
  "msg/PickeeVisionStaffLocation.msg"
  "msg/Point2D.msg"
  "msg/Point3D.msg"
  "msg/Pose2D.msg"
  "msg/ProductLocation.msg"
  "msg/Vector2D.msg"
)

set(srv_files
  "srv/MainGetProductLocation.srv"
  "srv/PackeeArmMoveToPose.srv"
  "srv/PackeeArmPickProduct.srv"
  "srv/PackeeArmPlaceProduct.srv"
  "srv/PackeePackingCheckAvailability.srv"
  "srv/PackeePackingStart.srv"
  "srv/PackeeVisionCheckCartPresence.srv"
  "srv/PackeeVisionDetectProductsInCart.srv"
  "srv/PackeeVisionVerifyPackingComplete.srv"
  "srv/PickeeArmMoveToPose.srv"
  "srv/PickeeArmPickProduct.srv"
  "srv/PickeeArmPlaceProduct.srv"
  "srv/PickeeMainVideoStreamStart.srv"
  "srv/PickeeMainVideoStreamStop.srv"
  "srv/PickeeMobileMoveToLocation.srv"
  "srv/PickeeMobileUpdateGlobalPath.srv"
  "srv/PickeeProductDetect.srv"
  "srv/PickeeProductProcessSelection.srv"
  "srv/PickeeTtsRequest.srv"
  "srv/PickeeVisionCheckCartPresence.srv"
  "srv/PickeeVisionCheckProductInCart.srv"
  "srv/PickeeVisionDetectProducts.srv"
  "srv/PickeeVisionRegisterStaff.srv"
  "srv/PickeeVisionSetMode.srv"
  "srv/PickeeVisionTrackStaff.srv"
  "srv/PickeeVisionVideoStreamStart.srv"
  "srv/PickeeVisionVideoStreamStop.srv"
  "srv/PickeeWorkflowEndShopping.srv"
  "srv/PickeeWorkflowMoveToPackaging.srv"
  "srv/PickeeWorkflowMoveToSection.srv"
  "srv/PickeeWorkflowReturnToBase.srv"
  "srv/PickeeWorkflowStartTask.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
