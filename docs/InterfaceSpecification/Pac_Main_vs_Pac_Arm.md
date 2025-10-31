Pac Main = Packee Main Controller

Pac Arm = Packee Arm Controller

### `/packee/arm/pose_status`
> **ROS2 Interface:** `shopee_interfaces/msg/ArmPoseStatus.msg`

### `/packee/arm/pick_status`
> **ROS2 Interface:** `shopee_interfaces/msg/ArmTaskStatus.msg`

### `/packee/arm/place_status`
> **ROS2 Interface:** `shopee_interfaces/msg/ArmTaskStatus.msg`

### `/packee/arm/move_to_pose`
> **ROS2 Interface:** `shopee_interfaces/srv/ArmMoveToPose.srv`

### `/packee/arm/pick_product`
> **ROS2 Interface:** `shopee_interfaces/srv/ArmPickProduct.srv`

### `/packee/arm/place_product`
> **ROS2 Interface:** `shopee_interfaces/srv/ArmPlaceProduct.srv`



## Topic 인터페이스

| 구분 | 메시지명 | 토픽 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **자세 변경 상태** | `/packee/arm/pose_status` | Topic | Pac Arm | Pac Main | `int32 robot_id`<br>`int32 order_id`<br>`string pose_type`<br>`string status`<br>`float32 progress`<br>`string message`<br><br>**status**<br>`"in_progress"` - 진행 중<br>`"complete"` - 완료<br>`"failed"` - 실패 | **진행 중**<br>`robot_id: 1`<br>`order_id: 3`<br>`pose_type: "cart_view"`<br>`status: "in_progress"`<br>`progress: 0.6`<br>`message: "Moving to cart view pose"`<br><br>**완료**<br>`robot_id: 1`<br>`order_id: 3`<br>`pose_type: "cart_view"`<br>`status: "complete"`<br>`progress: 1.0`<br>`message: "Reached cart view pose"` |
| **픽업 상태** | `/packee/arm/pick_status` | Topic | Pac Arm | Pac Main | `int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br>`string arm_side`<br>`string status`<br>`string current_phase`<br>`float32 progress`<br>`string message`<br><br>**arm_side**<br>`"left"` - 좌측 팔<br>`"right"` - 우측 팔<br><br>**status**<br>`"in_progress"` - 진행 중<br>`"completed"` - 완료<br>`"failed"` - 실패<br><br>**current_phase**<br>`"planning"` - 경로 계획 중<br>`"approaching"` - 접근 중<br>`"grasping"` - 그립 중<br>`"lifting"` - 들어올리는 중<br>`"done"` - 완료 | **좌측 팔 - 진행 중**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`arm_side: "left"`<br>`status: "in_progress"`<br>`current_phase: "grasping"`<br>`progress: 0.7`<br>`message: "Left arm grasping product"`<br><br>**우측 팔 - 완료**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`arm_side: "right"`<br>`status: "completed"`<br>`current_phase: "done"`<br>`progress: 1.0`<br>`message: "Right arm picked successfully"`<br><br>**좌측 팔 - 실패**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`arm_side: "left"`<br>`status: "failed"`<br>`current_phase: "grasping"`<br>`progress: 0.7`<br>`message: "Left arm grasp failed - gripper error"` |
| **담기 상태** | `/packee/arm/place_status` | Topic | Pac Arm | Pac Main | `int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br>`string arm_side`<br>`string status`<br>`string current_phase`<br>`float32 progress`<br>`string message`<br><br>**arm_side**<br>`"left"` - 좌측 팔<br>`"right"` - 우측 팔<br><br>**status**<br>`"in_progress"` - 진행 중<br>`"completed"` - 완료<br>`"failed"` - 실패<br><br>**current_phase**<br>`"planning"` - 경로 계획 중<br>`"approaching"` - 접근 중<br>`"moving"` - 포장 위치 정렬/복귀 중<br>`"done"` - 완료 | **우측 팔 - 진행 중**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`arm_side: "right"`<br>`status: "in_progress"`<br>`current_phase: "moving"`<br>`progress: 0.5`<br>`message: "Right arm moving to packing box"`<br><br>**좌측 팔 - 완료**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`arm_side: "left"`<br>`status: "completed"`<br>`current_phase: "done"`<br>`progress: 1.0`<br>`message: "Left arm placed successfully"`<br><br>**우측 팔 - 실패**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`arm_side: "right"`<br>`status: "failed"`<br>`current_phase: "planning"`<br>`progress: 0.1`<br>`message: "Right arm path planning failed - collision detected"` |

## Service 인터페이스

| 구분 | 서비스명 | 서비스 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **자세 변경 명령** | `/packee/arm/move_to_pose` | Service | Pac Main | Pac Arm | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`string pose_type`<br><br>**Response**<br>`bool success`<br>`string message`<br><br>**pose_type**<br>`"cart_view"` - 장바구니 확인 자세<br>`"standby"` - 대기 자세 | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br>`pose_type: "cart_view"`<br><br>**Response**<br>`success: true`<br>`message: "Pose change command accepted"` |
| **상품 픽업 명령** | `/packee/arm/pick_product` | Service | Pac Main | Pac Arm | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`string arm_side`<br>`shopee_interfaces/msg/DetectedProduct target_product`<br><br>**DetectedProduct 사용 규칙 (Packee)**<br>`int32 product_id`<br>`float32 confidence`<br>`shopee_interfaces/msg/BBox bbox`<br>`shopee_interfaces/msg/Pose6D pose`<br>`int32 bbox_number` (0 고정)<br>`shopee_interfaces/msg/DetectionInfo detection_info` (빈 배열)<br><br>**Response**<br>`bool success`<br>`string message`<br><br>**arm_side**<br>`"left"` - 좌측 팔 사용<br>`"right"` - 우측 팔 사용 | **Request - 좌측 팔**<br>`robot_id: 1`<br>`order_id: 3`<br>`arm_side: "left"`<br>`target_product:`<br>`  product_id: 3`<br>`  bbox: {x1: 120, y1: 180, x2: 250, y2: 320}`<br>`  position: {x: 0.3, y: 0.15, z: 0.8}`<br>`  confidence: 0.94`<br><br>**Response**<br>`success: true`<br>`message: "Left arm pick command accepted"`<br><br>**Request - 우측 팔**<br>`robot_id: 1`<br>`order_id: 3`<br>`arm_side: "right"`<br>`target_product:`<br>`  product_id: 3`<br>`  bbox: {x1: 280, y1: 150, x2: 380, y2: 280}`<br>`  pose: {x: 0.25, y: -0.1, z: 0.75, rx: 0.3, ry: 0.35, rz: 0.66}`<br>`  confidence: 0.96`<br><br>**Response**<br>`success: true`<br>`message: "Right arm pick command accepted"` |
| **상품 담기 명령** | `/packee/arm/place_product` | Service | Pac Main | Pac Arm | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br>`string arm_side`<br>`shopee_interfaces/msg/Pose6D pose`<br><br>**Response**<br>`bool success`<br>`string message`<br><br>**arm_side**<br>`"left"` - 좌측 팔 사용<br>`"right"` - 우측 팔 사용 | **Request - 좌측 팔**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`arm_side: "left"`<br>`pose: {x: 0.25, y: -0.1, z: 0.75, rx: 0.3, ry: 0.35, rz: 0.66}`<br><br>**Response**<br>`success: true`<br>`message: "Left arm place command accepted"`<br><br>**Request - 우측 팔**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`arm_side: "right"`<br>`pose: {x: 0.25, y: -0.1, z: 0.75, rx: 0.3, ry: 0.35, rz: 0.66}`<br><br>**Response**<br>`success: true`<br>`message: "Right arm place command accepted"` |
