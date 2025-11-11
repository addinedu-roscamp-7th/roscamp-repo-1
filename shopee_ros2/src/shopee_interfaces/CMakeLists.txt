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
  "msg/ArmTaskStatus.msg"
  "msg/ArucoPose.msg"
  "msg/BBox.msg"
  "msg/DetectedProduct.msg"
  "msg/DetectionInfo.msg"
  "msg/Obstacle.msg"
  "msg/PackeeAvailability.msg"
  "msg/PackeePackingComplete.msg"
  "msg/PackeeRobotStatus.msg"
  "msg/PersonDetection.msg"
  "msg/PickeeArrival.msg"
  "msg/PickeeCartHandover.msg"
  "msg/PickeeMobileArrival.msg"
  "msg/PickeeMobilePose.msg"
  "msg/PickeeMobileSpeedControl.msg"
  "msg/PickeeMoveStatus.msg"
  "msg/PickeeProductDetection.msg"
  "msg/PickeeProductLoaded.msg"
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
  "msg/Pose6D.msg"
  "msg/ProductLocation.msg"
  "msg/ProductInfo.msg"
  "msg/Vector2D.msg"
  "msg/Sequence.msg"
)

set(srv_files
  "srv/ArmMoveToPose.srv"
  "srv/ArmPickProduct.srv"
  "srv/ArmPlaceProduct.srv"
  "srv/ChangeTrackingMode.srv"
  "srv/MainGetAvailableRobots.srv"
  "srv/MainGetProductLocation.srv"
  "srv/MainGetLocationPose.srv"
  "srv/MainGetWarehousePose.srv"
  "srv/MainGetSectionPose.srv"
  "srv/PackeePackingCheckAvailability.srv"
  "srv/PackeePackingStart.srv"
  "srv/PackeeVisionDetectProductsInCart.srv"
  "srv/PackeeVisionVerifyPackingComplete.srv"
  "srv/PickeeMainVideoStreamStart.srv"
  "srv/PickeeMainVideoStreamStop.srv"
  "srv/PickeeMobileMoveToLocation.srv"
  "srv/PickeeMobileUpdateGlobalPath.srv"
  "srv/PickeeProductDetect.srv"
  "srv/PickeeProductProcessSelection.srv"
  "srv/PickeeTtsRequest.srv"
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
  "srv/PickeeWorkflowReturnToStaff.srv"
  "srv/PickeeWorkflowStartTask.srv"
  "srv/VisionCheckCartPresence.srv"
  "srv/PackeeArmPackingComplete.srv"
  "srv/PackeeMainStartMTC.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
