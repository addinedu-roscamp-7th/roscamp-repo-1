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

> myCobot 280의 안전 작업 공간은 수평 반경 0.28 m, Z 0.05~0.30 m 입니다. 아래 테스트 값도 이 범위 안에서 설정해야 합니다.

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
        x: 0.12,
        y: -0.05,
        z: 0.18
    },
    bbox: {
        x1: 220,
        y1: 140,
        x2: 360,
        y2: 320
    }
}"
ros2 service call /packee/arm/place_product shopee_interfaces/srv/PackeeArmPlaceProduct "{
    robot_id: 1,
    order_id: 10,
    product_id: 20,
    arm_side: 'left',
    box_position: {
        x: 0.10,
        y: 0.06,
        z: 0.16
    }
}"
```
### 토픽
``` bash
ros2 topic echo /packee/arm/pick_status
ros2 topic echo /packee/arm/place_status
ros2 topic echo /packee/arm/pose_status


ros2 topic pub --once /packee/arm/pose_status shopee_interfaces/msg/ArmPoseStatus "{robot_id: 1, order_id: 123, pose_type: 'cart_view', status: 'test', progress: 0.5, message: 'success'}"
ros2 topic pub --once /packee/arm/place_status shopee_interfaces/msg/PackeeArmTaskStatus "{robot_id: 1, order_id: 123, product_id: 10, arm_side: 'left', status: 'running', current_phase: 'pick', progress: 0.6, message: 'object grasped'}"
ros2 topic pub --once /packee/arm/pick_status shopee_interfaces/msg/PackeeArmTaskStatus "{robot_id: 1, order_id: 123, product_id: 10, arm_side: 'left', status: 'running', current_phase: 'pick', progress: 0.6, message: 'object grasped'}"


```
