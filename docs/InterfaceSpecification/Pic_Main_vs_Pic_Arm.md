Pic Main = Pickee Main Controller

Pic Arm = Pickee Arm Controller

### `/pickee/arm/pose_status`
> **ROS2 Interface:** `shopee_interfaces/msg/ArmPoseStatus.msg`

### `/pickee/arm/pick_status`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeArmTaskStatus.msg`

### `/pickee/arm/place_status`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeArmTaskStatus.msg`

### `/pickee/arm/move_to_pose`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeArmMoveToPose.srv`

### `/pickee/arm/pick_product`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeArmPickProduct.srv`

### `/pickee/arm/place_product`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeArmPlaceProduct.srv`

## Topic 인터페이스

| 구분 | 메시지명 | 토픽 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **자세 변경 상태** | `/pickee/arm/pose_status` | Topic | Pic Arm | Pic Main | `int32 robot_id`<br>`int32 order_id`<br>`string pose_type`<br>`string status`<br>`float32 progress`<br>`string message`<br><br>**status**<br>`"in_progress"` - 진행 중<br>`"completed"` - 완료<br>`"failed"` - 실패 | **진행 중**<br>`robot_id: 1`<br>`order_id: 3`<br>`pose_type: "shelf_view"`<br>`status: "in_progress"`<br>`progress: 0.6`<br>`message: "Moving to shelf view pose"`<br><br>**완료**<br>`robot_id: 1`<br>`order_id: 4`<br>`pose_type: "shelf_view"`<br>`status: "completed"`<br>`progress: 1.0`<br>`message: "Reached shelf view pose"`<br><br>**실패**<br>`robot_id: 1`<br>`order_id: 5`<br>`pose_type: "cart_view"`<br>`status: "failed"`<br>`progress: 0.3`<br>`message: "Joint limit exceeded"` |
| **픽업 상태** | `/pickee/arm/pick_status` | Topic | Pic Arm | Pic Main | `int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br>`string status`<br>`string current_phase`<br>`float32 progress`<br>`string message`<br><br>**status**<br>`"in_progress"` - 진행 중<br>`"completed"` - 완료<br>`"failed"` - 실패<br><br>**current_phase**<br>`"planning"` - 경로 계획 중<br>`"approaching"` - 접근 중<br>`"grasping"` - 그립 중<br>`"lifting"` - 들어올리는 중<br>`"done"` - 완료 | **진행 중 - 경로 계획**<br>`robot_id: 1`<br>`order_id: 4`<br>`product_id: 5`<br>`status: "in_progress"`<br>`current_phase: "planning"`<br>`progress: 0.2`<br>`message: "Planning grasp trajectory"`<br><br>**진행 중 - 그립**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 5`<br>`status: "in_progress"`<br>`current_phase: "grasping"`<br>`progress: 0.7`<br>`message: "Grasping product"`<br><br>**완료**<br>`robot_id: 1`<br>`order_id: 23`<br>`product_id: 32`<br>`status: "completed"`<br>`current_phase: "done"`<br>`progress: 1.0`<br>`message: "Product picked successfully"`<br><br>**실패**<br>`robot_id: 1`<br>`order_id: 6`<br>`product_id: 7`<br>`status: "failed"`<br>`current_phase: "grasping"`<br>`progress: 0.7`<br>`message: "Grasp failed - gripper error"` |
| **담기 상태** | `/pickee/arm/place_status` | Topic | Pic Arm | Pic Main | `int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br>`string status`<br>`string current_phase`<br>`float32 progress`<br>`string message`<br><br>**status**<br>`"in_progress"` - 진행 중<br>`"completed"` - 완료<br>`"failed"` - 실패<br><br>**current_phase**<br>`"planning"` - 경로 계획 중<br>`"moving"` - 이동 중<br>`"placing"` - 놓는 중<br>`"releasing"` - 그립 해제 중<br>`"done"` - 완료 | **진행 중 - 이동**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`status: "in_progress"`<br>`current_phase: "moving"`<br>`progress: 0.5`<br>`message: "Moving to cart"`<br><br>**완료**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`status: "completed"`<br>`current_phase: "done"`<br>`progress: 1.0`<br>`message: "Product placed in cart successfully"`<br><br>**실패**<br>`robot_id: 1`<br>`order_id: 3`<br>`product_id: 3`<br>`status: "failed"`<br>`current_phase: "planning"`<br>`progress: 0.1`<br>`message: "Path planning failed - collision detected"` |

## Service 인터페이스

| 구분 | 서비스명 | 서비스 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **자세 변경 요청** | `/pickee/arm/move_to_pose` | Service | Pic Main | Pic Arm | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`string pose_type`<br><br>**Response**<br>`bool success`<br>`string message`<br><br>**pose_type**<br>`"shelf_view"` - 매대 확인 자세<br>`"cart_view"` - 장바구니 확인 자세<br>`"standby"` - 대기 자세 | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br>`pose_type: "shelf_view"`<br><br>**Response**<br>`success: true`<br>`message: "Pose change command accepted"` |
| **상품 픽업 요청** | `/pickee/arm/pick_product` | Service | Pic Main | Pic Arm | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`DetectedProduct target_product`<br><br>**Response**<br>`bool accepted`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br>`target_product: {`<br>`  product_id: 4`<br>`  bbox_number: 1`<br>`  bbox_coords: {x1: 100, y1: 150, x2: 200, y2: 250}`<br>`  confidence: 0.95`<br>`}`<br><br>**Response**<br>`accepted: true`<br>`message: "Pick command accepted"` |
| **상품 담기 요청** | `/pickee/arm/place_product` | Service | Pic Main | Pic Arm | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br><br>**Response**<br>`bool accepted`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 21`<br>`product_id: 34`<br><br>**Response**<br>`accepted: true`<br>`message: "Place command accepted"` |