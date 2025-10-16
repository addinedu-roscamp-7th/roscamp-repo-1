# Pac Arm 테스트 가이드
## 서비스 & 토픽 
### 서비스 리스트 : ros2 service list -t
``` bash
service /packee/arm/move_to_pose [shopee_interfaces/srv/PackeeArmMoveToPose]
service /packee/arm/pick_product [shopee_interfaces/srv/PackeeArmPickProduct]
service /packee/arm/place_product [shopee_interfaces/srv/PackeeArmPlaceProduct]
```

### 토픽 리스트 : ros2 topic list -t
``` bash
pub /packee/arm/pick_status [shopee_interfaces/msg/PackeeArmTaskStatus]
pub /packee/arm/place_status [shopee_interfaces/msg/PackeeArmTaskStatus]
pub /packee/arm/pose_status [shopee_interfaces/msg/ArmPoseStatus]
sub /packee/availability_result [shopee_interfaces/msg/PackeeAvailability]
sub /packee/packing_complete [shopee_interfaces/msg/PackeePackingComplete]
sub /packee/robot_status [shopee_interfaces/msg/PackeeRobotStatus]
```

## 명령어 예시
### 서비스
``` bash
ros2 service call /packee/arm/move_to_pose shopee_interfaces/srv/PackeeArmMoveToPose "{}"
```
### 토픽
``` bash
ros2 topic echo /packee/arm/pick_status
ros2 topic echo /packee/arm/place_status
ros2 topic echo /packee/arm/pose_status
```

