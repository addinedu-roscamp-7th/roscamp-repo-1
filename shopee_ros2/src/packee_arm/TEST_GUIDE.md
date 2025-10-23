# Pac Arm 테스트 가이드
## 서비스 & 토픽 
### 서비스 리스트 : ros2 service list -t
``` bash
service /packee/arm/move_to_pose [shopee_interfaces/srv/ArmMoveToPose]
service /packee/arm/pick_product [shopee_interfaces/srv/ArmPickProduct]
service /packee/arm/place_product [shopee_interfaces/srv/ArmPlaceProduct]
```

### 토픽 리스트 : ros2 topic list -t
``` bash
pub /packee/arm/pick_status [shopee_interfaces/msg/ArmTaskStatus]
pub /packee/arm/place_status [shopee_interfaces/msg/ArmTaskStatus]
pub /packee/arm/pose_status [shopee_interfaces/msg/ArmPoseStatus]
sub /packee/availability_result [shopee_interfaces/msg/PackeeAvailability]
sub /packee/packing_complete [shopee_interfaces/msg/PackeePackingComplete]
sub /packee/robot_status [shopee_interfaces/msg/PackeeRobotStatus]
```

`ArmPickProduct` 요청 본문은 `DetectedProduct` 메시지를 그대로 중첩합니다.  
Packee 기준 필수 필드는 `product_id`, `confidence`, `bbox`, `pose`, `bbox_number(=0)`, `detection_info`입니다.  
`pose`는 joint_1~joint_6을 사용하며, 테스트 시 joint_5~6은 0으로 두어도 됩니다.

> myCobot 280의 안전 작업 공간은 수평 반경 0.28 m, Z 0.05~0.30 m 입니다. 아래 테스트 값도 이 범위 안에서 설정해야 합니다.

## 명령어 예시
### 서비스
``` bash
ros2 service call /packee/arm/move_to_pose shopee_interfaces/srv/ArmMoveToPose "{
    robot_id: 1,
    order_id: 10,
    pose_type: 'cart_view'
}"
ros2 service call /packee/arm/pick_product shopee_interfaces/srv/ArmPickProduct "{
    robot_id: 1,
    order_id: 10,
    arm_side: 'left',
    target_product: {
        product_id: 20,
        confidence: 0.93,
        bbox: {
            x1: 220,
            y1: 140,
            x2: 360,
            y2: 320
        },
        bbox_number: 0,
        detection_info: {
            polygon: [],
            bbox_coords: {x1: 0, y1: 0, x2: 0, y2: 0}
        },
        pose: {
            joint_1: 0.12,
            joint_2: -0.05,
            joint_3: 0.18,
            joint_4: 0.0,
            joint_5: 0.0,
            joint_6: 0.0
        }
    }
}"
ros2 service call /packee/arm/place_product shopee_interfaces/srv/ArmPlaceProduct "{
    robot_id: 1,
    order_id: 10,
    product_id: 20,
    arm_side: 'left',
    pose: {
        joint_1: 0.10,
        joint_2: 0.06,
        joint_3: 0.16,
        joint_4: 0.0,
        joint_5: 0.0,
        joint_6: 0.0
    }
}"
```
### 토픽
``` bash
ros2 topic echo /packee/arm/pick_status
ros2 topic echo /packee/arm/place_status
ros2 topic echo /packee/arm/pose_status


ros2 topic pub --once /packee/arm/pose_status shopee_interfaces/msg/ArmPoseStatus "{robot_id: 1, order_id: 123, pose_type: 'cart_view', status: 'test', progress: 0.5, message: 'success'}"
ros2 topic pub --once /packee/arm/place_status shopee_interfaces/msg/ArmTaskStatus "{robot_id: 1, order_id: 123, product_id: 10, arm_side: 'left', status: 'in_progress', current_phase: 'moving', progress: 0.6, message: 'object moving to box'}"
ros2 topic pub --once /packee/arm/pick_status shopee_interfaces/msg/ArmTaskStatus "{robot_id: 1, order_id: 123, product_id: 10, arm_side: 'left', status: 'in_progress', current_phase: 'grasping', progress: 0.6, message: 'object grasped'}"


```
