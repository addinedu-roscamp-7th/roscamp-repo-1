Pic Main = Pickee Main Controller

Pic Arm = Pickee Arm Controller

## ROS 인터페이스 매핑

| 분류 | 토픽/서비스 | ROS 타입 |
|---|---|---|
| Topic | `/pickee/arm/pose_status` | `shopee_interfaces/msg/ArmPoseStatus` |
| Topic | `/pickee/arm/pick_status` | `shopee_interfaces/msg/PickeeArmTaskStatus` |
| Topic | `/pickee/arm/place_status` | `shopee_interfaces/msg/PickeeArmTaskStatus` |
| Service | `/pickee/arm/move_to_pose` | `shopee_interfaces/srv/PickeeArmMoveToPose` |
| Service | `/pickee/arm/pick_product` | `shopee_interfaces/srv/PickeeArmPickProduct` |
| Service | `/pickee/arm/place_product` | `shopee_interfaces/srv/PickeeArmPlaceProduct` |

 
 
 

From

To

Message

예시

Topic

 

 

 

 

 

자세 변경 상태

/pickee/arm/pose_status

Pic Arm

Pic Main



int32 robot_id
int32 order_id
string pose_type
string status
float32 progress
string message
status:

"in_progress" - 진행 중

"completed" - 완료

"failed" - 실패

 

 



# 진행 중
robot_id: 1
order_id: 3
pose_type: "shelf_view"
status: "in_progress"
progress: 0.6
message: "Moving to shelf view pose"
# 완료
robot_id: 1
order_id: 4
pose_type: "shelf_view"
status: "completed"
progress: 1.0
message: "Reached shelf view pose"
# 실패
robot_id: 1
order_id: 5
pose_type: "cart_view"
status: "failed"
progress: 0.3
message: "Joint limit exceeded"
픽업 상태

/pickee/arm/pick_status

Pic Arm

Pic Main



int32 robot_id
int32 order_id
int32 product_id
string status
string current_phase
float32 progress
string message
status

"in_progress" - 진행 중

"completed" - 완료

"failed" - 실패

current_phase

"planning" - 경로 계획 중

"approaching" - 접근 중

"grasping" - 그립 중

"lifting" - 들어올리는 중

"done" - 완료

 



# 진행 중 - 경로 계획
robot_id: 1
order_id: 4
product_id: 5
status: "in_progress"
current_phase: "planning"
progress: 0.2
message: "Planning grasp trajectory"
# 진행 중 - 그립
robot_id: 1
order_id: 3
product_id: 5
status: "in_progress"
current_phase: "grasping"
progress: 0.7
message: "Grasping product"
# 완료
robot_id: 1
order_id: 23
product_id: 32
status: "completed"
current_phase: "done"
progress: 1.0
message: "Product picked successfully"
# 실패
robot_id: 1
order_id: 6
product_id: 7
status: "failed"
current_phase: "grasping"
progress: 0.7
message: "Grasp failed - gripper error"
담기 상태

/pickee/arm/place_status

Pic Arm

Pic Main



int32 robot_id
int32 order_id
int32 product_id
string status
string current_phase
float32 progress
string message
status

"in_progress" - 진행 중

"completed" - 완료

"failed" - 실패

current_phase

"planning" - 경로 계획 중

"moving" - 이동 중

"placing" - 놓는 중

"releasing" - 그립 해제 중

"done" - 완료



# 진행 중 - 이동
robot_id: 1
order_id: 3
product_id: 3
status: "in_progress"
current_phase: "moving"
progress: 0.5
message: "Moving to cart"
# 완료
robot_id: 1
order_id: 3
product_id: 3
status: "completed"
current_phase: "done"
progress: 1.0
message: "Product placed in cart successfully"
# 실패
robot_id: 1
order_id: 3
product_id: 3
status: "failed"
current_phase: "planning"
progress: 0.1
message: "Path planning failed - collision detected"
Service

 

 

 

 

 

자세 변경 요청

/pickee/arm/move_to_pose

Pic Main

Pic Arm



# Request
int32 robot_id
int32 order_id
string pose_type
---
# Response
bool success
string message
pose_type

"shelf_view" - 매대 확인 자세

"cart_view" - 장바구니 확인 자세

"standby" - 대기 자세



# Request
robot_id: 1
order_id: 3
pose_type: "shelf_view"
# Response
success: true
message: "Pose change command accepted"
상품 픽업 요청

/pickee/arm/pick_product

Pic Main

Pic Arm



# Request
int32 robot_id
int32 order_id
int32 product_id
Point3D target_position
---
# Response
bool accepted
string message


# Request
robot_id: 1
order_id: 3
product_id: 4
target_position: {x: 0.5, y: 0.3, z: 1.2}
# Response
accepted: true
message: "Pick command accepted"
상품 담기 요청

/pickee/arm/place_product

Pic Main

Pic Arm



# Request
int32 robot_id
int32 order_id
int32 product_id
---
# Response
bool accepted
string message


# Request
robot_id: 1
order_id: 21
product_id: 34
# Response
accepted: true
message: "Place command accepted"
