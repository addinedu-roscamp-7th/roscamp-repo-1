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

int32 robot_id
int32 order_id
int32 product_id
string arm_side
shopee_interfaces/Point3D target_position
shopee_interfaces/BBox bbox

## 명령어 예시
### 서비스
``` bash
ros2 service call /packee/arm/move_to_pose shopee_interfaces/srv/PackeeArmMoveToPose "{
    robot_id: 1,
    order_id: 10,
    pose_type: "cart_view"
}"
ros2 service call /packee/arm/pick_product shopee_interfaces/srv/PackeeArmPickProduct "{
    robot_id: 1,
    order_id: 10,
    product_id: 20,
    arm_side: 'left',
    target_position: {
        x: 1.0,
        y: 1.0,
        z: 1.0
    }
    bbox: {
        x1: 1.0,
        y1: 1.0,
        x2: 1.0,
        y2: 1.0
    }
}"
```
### 토픽
``` bash
ros2 topic echo /packee/arm/pick_status
ros2 topic echo /packee/arm/place_status
ros2 topic echo /packee/arm/pose_status
```

