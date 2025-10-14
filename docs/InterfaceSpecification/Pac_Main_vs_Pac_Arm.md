Pac Main = Packee Main Controller

Pac Arm = Packee Arm Controller

### `/packee/arm/pose_status`
> **ROS2 Interface:** `shopee_interfaces/msg/ArmPoseStatus.msg`

### `/packee/arm/pick_status`
> **ROS2 Interface:** `shopee_interfaces/msg/PackeeArmTaskStatus.msg`

### `/packee/arm/place_status`
> **ROS2 Interface:** `shopee_interfaces/msg/PackeeArmTaskStatus.msg`

### `/packee/arm/move_to_pose`
> **ROS2 Interface:** `shopee_interfaces/srv/PackeeArmMoveToPose.srv`

### `/packee/arm/pick_product`
> **ROS2 Interface:** `shopee_interfaces/srv/PackeeArmPickProduct.srv`

### `/packee/arm/place_product`
> **ROS2 Interface:** `shopee_interfaces/srv/PackeeArmPlaceProduct.srv`



From

To

Message

예시

Topic











자세 변경 상태

/packee/arm/pose_status

Pac Arm

Pac Main

int32 robot_id
int32 order_id
string pose_type
string status
float32 progress
string message

status

"in_progress" - 진행 중

"completed" - 완료

"failed" - 실패



# 진행 중
robot_id: 1
order_id: 3
pose_type: "cart_view"
status: "in_progress"
progress: 0.6
message: "Moving to cart view pose"

# 완료
robot_id: 1
order_id: 3
pose_type: "cart_view"
status: "completed"
progress: 1.0
message: "Reached cart view pose"

픽업 상태

/packee/arm/pick_status

Pac Arm

Pac Main

int32 robot_id
int32 order_id
int32 product_id
string arm_side
string status
string current_phase
float32 progress
string message

arm_side

"left" - 좌측 팔

"right" - 우측 팔

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



# 좌측 팔 - 진행 중
robot_id: 1
order_id: 3
product_id: 3
arm_side: "left"
status: "in_progress"
current_phase: "grasping"
progress: 0.7
message: "Left arm grasping product"

# 우측 팔 - 완료
robot_id: 1
order_id: 3
product_id: 3
arm_side: "right"
status: "completed"
current_phase: "done"
progress: 1.0
message: "Right arm picked successfully"

# 좌측 팔 - 실패
robot_id: 1
order_id: 3
product_id: 3
arm_side: "left"
status: "failed"
current_phase: "grasping"
progress: 0.7
message: "Left arm grasp failed - gripper error"

담기 상태

/packee/arm/place_status

Pac Arm

Pac Main

int32 robot_id
int32 order_id
int32 product_id
string arm_side
string status
string current_phase
float32 progress
string message

arm_side

"left" - 좌측 팔

"right" - 우측 팔

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

# 우측 팔 - 진행 중
robot_id: 1
order_id: 3
product_id: 3
arm_side: "right"
status: "in_progress"
current_phase: "moving"
progress: 0.5
message: "Right arm moving to packing box"

# 좌측 팔 - 완료
robot_id: 1
order_id: 3
product_id: 3
arm_side: "left"
status: "completed"
current_phase: "done"
progress: 1.0
message: "Left arm placed successfully"

# 우측 팔 - 실패
robot_id: 1
order_id: 3
product_id: 3
arm_side: "right"
status: "failed"
current_phase: "planning"
progress: 0.1
message: "Right arm path planning failed - collision detected"

Service











자세 변경 명령

/packee/arm/move_to_pose

Pac Main

Pac Arm

# Request
int32 robot_id
int32 order_id
string pose_type
---
# Response
bool accepted
string message

pose_type

"cart_view" - 장바구니 확인 자세

"standby" - 대기 자세

# Request
robot_id: 1
order_id: 3
pose_type: "cart_view"

# Response
accepted: true
message: "Pose change command accepted"

상품 픽업 명령

/packee/arm/pick_product

Pac Main

Pac Arm

# Request
int32 robot_id
int32 order_id
int32 product_id
string arm_side
Point3D target_position
BBox bbox
---
# Response
bool accepted
string message

arm_side

"left" - 좌측 팔 사용

"right" - 우측 팔 사용



# Request - 좌측 팔
robot_id: 1
order_id: 3
product_id: 3
arm_side: "left"
target_position: {x: 0.3, y: 0.15, z: 0.8}
bbox: {x1: 120, y1: 180, x2: 250, y2: 320}

# Response
accepted: true
message: "Left arm pick command accepted"

---

# Request - 우측 팔
robot_id: 1
order_id: 3
product_id: 3
arm_side: "right"
target_position: {x: 0.25, y: -0.1, z: 0.75}
bbox: {x1: 280, y1: 150, x2: 380, y2: 280}

# Response
accepted: true
message: "Right arm pick command accepted"

상품 담기 명령

/packee/arm/place_product

Pac Main

Pac Arm

# Request
int32 robot_id
int32 order_id
int32 product_id
string arm_side
Point3D box_position
---
# Response
bool accepted
string message

arm_side

"left" - 좌측 팔 사용

"right" - 우측 팔 사용



# Request - 좌측 팔
robot_id: 1
order_id: 3
product_id: 3
arm_side: "left"
box_position: {x: 0.5, y: 0.3, z: 0.2}

# Response
accepted: true
message: "Left arm place command accepted"

---

# Request - 우측 팔
robot_id: 1
order_id: 3
product_id: 3
arm_side: "right"
box_position: {x: 0.5, y: -0.3, z: 0.2}

# Response
accepted: true
message: "Right arm place command accepted"
