# Shopee Interfaces

Shopee 로봇 프로젝트 전반에서 사용하는 ROS 2 메시지·서비스 인터페이스 패키지입니다.  
`docs/InterfaceSpecification/*.md` 명세서를 기준으로 Pickee / Packee / Main 컴포넌트 간 통신 규약을 정리했습니다.

## 메시지 구성
- **공통 기본 타입**: `Point2D`, `Point3D`, `Pose2D`, `Vector2D`, `BBox`, `Obstacle`, `ProductLocation`
- **팔 상태**: `ArmPoseStatus`, `ArmTaskStatus`
- **모바일 베이스**: `PickeeMobilePose`, `PickeeMobileArrival`, `PickeeMobileSpeedControl`
- **비전 이벤트**: `PickeeVisionDetection`, `PickeeVisionObstacles`, `PackeeDetectedProduct`, `PickeeDetectedProduct`
- **메인 서비스 연계**: `PickeeMoveStatus`, `PickeeArrival`, `PickeeRobotStatus`, `PackeePackingComplete` 등

## 서비스 구성
- **픽업/포장 제어**: `Arm*`, `PickeeArm*`, `PackeePacking*`
- **비전 연동**: `PickeeVision*`, `PackeeVision*`, `PickeeTtsRequest`
- **모바일 주행**: `PickeeMobileMoveToLocation`, `PickeeMobileUpdateGlobalPath`
- **워크플로 관리**: `PickeeWorkflow*`, `PickeeProduct*`, `MainGetProductLocation`
- **영상 스트리밍**: `PickeeMainVideoStream*`, `PickeeVisionVideoStream*`

## 사용 방법
1. 워크스페이스 루트에서 인터페이스 코드를 생성합니다.
   ```bash
   cd /home/jinhyuk2me/dev_ws/Shopee/ros2_ws
   colcon build --packages-select shopee_interfaces
   ```
2. 새 터미널에서 `source install/setup.bash` 실행 후 메시지/서비스 타입을 사용할 수 있습니다.

## 참고 문서
- `docs/InterfaceSpecification/Pic_Main_vs_Pic_*`
- `docs/InterfaceSpecification/Pac_Main_vs_Pac_*`
- `docs/InterfaceSpecification/Main_vs_*`
