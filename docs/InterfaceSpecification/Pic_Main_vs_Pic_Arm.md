Pic Main = Pickee Main Controller

Pic Arm = Pickee Arm Controller

### `/pickee/arm/pose_status`
> **ROS2 Interface:** `shopee_interfaces/msg/ArmPoseStatus.msg`

### `/pickee/arm/pick_status`
> **ROS2 Interface:** `shopee_interfaces/msg/ArmTaskStatus.msg`

### `/pickee/arm/place_status`
> **ROS2 Interface:** `shopee_interfaces/msg/ArmTaskStatus.msg`

### `/pickee/arm/move_to_pose`
> **ROS2 Interface:** `shopee_interfaces/srv/ArmMoveToPose.srv`

### `/pickee/arm/pick_product`
> **ROS2 Interface:** `shopee_interfaces/srv/ArmPickProduct.srv`

### `/pickee/arm/place_product`
> **ROS2 Interface:** `shopee_interfaces/srv/ArmPlaceProduct.srv`

## 🤖 인터페이스 상세 정의

## 📦 메시지 (Messages)

---

### 🧍 자세 변경 상태  
- **Topic**: `/pickee/arm/pose_status`  
- **From → To**: Pic Arm → Pic Main  
- **Message Fields**:
```plaintext
int32 robot_id
int32 order_id
string pose_type       # "shelf_view", "cart_view", "standby"
string status          # "in_progress", "completed", "failed"
float32 progress       # 0.0 ~ 1.0
string message
```

#### 예시:
- 진행 중:
```plaintext
pose_type: "shelf_view"
status: "in_progress"
progress: 0.6
message: "Moving to shelf view pose"
```
- 완료:
```plaintext
status: "completed"
progress: 1.0
message: "Reached shelf view pose"
```
- 실패:
```plaintext
status: "failed"
progress: 0.3
message: "Joint limit exceeded"
```

---

### ✋ 픽업 상태  
- **Topic**: `/pickee/arm/pick_status`  
- **From → To**: Pic Arm → Pic Main  
- **Message Fields**:
```plaintext
int32 robot_id
int32 order_id
int32 product_id
string arm_side        # Pickee는 ""로 송신
string status          # "in_progress", "completed", "failed"
string current_phase   # "planning", "approaching", "grasping", "lifting", "done"
float32 progress       # 0.0 ~ 1.0
string message
```

#### 예시:
- 경로 계획 중:
```plaintext
status: "in_progress"
current_phase: "planning"
progress: 0.2
message: "Planning grasp trajectory"
```
- 그립 중:
```plaintext
status: "in_progress"
current_phase: "grasping"
progress: 0.7
message: "Grasping product"
```
- 완료:
```plaintext
status: "completed"
current_phase: "done"
progress: 1.0
message: "Product picked successfully"
```
- 실패:
```plaintext
status: "failed"
current_phase: "grasping"
progress: 0.7
message: "Grasp failed - gripper error"
```

---

### 📥 담기 상태  
- **Topic**: `/pickee/arm/place_status`  
- **From → To**: Pic Arm → Pic Main  
- **Message Fields**:
```plaintext
int32 robot_id
int32 order_id
int32 product_id
string arm_side        # Pickee는 ""로 송신
string status          # "in_progress", "completed", "failed"
string current_phase   # "planning", "moving", "placing", "releasing", "done"
float32 progress
string message
```

> ※ Packee와의 공통 규격으로 `arm_side`가 포함되며 Pickee는 빈 문자열을 유지한다.

#### 예시:
- 이동 중:
```plaintext
status: "in_progress"
current_phase: "moving"
progress: 0.5
message: "Moving to cart"
```
- 완료:
```plaintext
status: "completed"
current_phase: "done"
progress: 1.0
message: "Product placed in cart successfully"
```
- 실패:
```plaintext
status: "failed"
current_phase: "planning"
progress: 0.1
message: "Path planning failed - collision detected"
```

---

## 🛠️ 서비스 (Services)

---

### 🤖 자세 변경 요청  
- **Service**: `/pickee/arm/move_to_pose`  
- **From → To**: Pic Main → Pic Arm

#### Request:
```plaintext
int32 robot_id
int32 order_id
string pose_type       # "shelf_view", "cart_view", "standby"
```

#### Response:
```plaintext
bool success
string message
```

#### 예시:
```plaintext
Request:
robot_id: 1
order_id: 3
pose_type: "shelf_view"

Response:
success: true
message: "Pose change command accepted"
```

---

### 🛒 상품 픽업 요청  
- **Service**: `/pickee/arm/pick_product`  
- **From → To**: Pic Main → Pic Arm

#### Request:
```plaintext
int32 robot_id
int32 order_id
string arm_side                 # Pickee는 "" 사용
shopee_interfaces/msg/Pose6D pose
```

- **Pose6D**
```plaintext
float32 x
float32 y
float32 z
float32 rx
float32 ry
float32 rz
```

#### Response:
```plaintext
bool success
string message
```

#### 예시:
```plaintext
Request:
  robot_id: 1
  order_id: 3
  arm_side: ""
  target_product: {
    product_id: 4
    bbox_number: 1
    detection_info: {
      polygon: [...]
      bbox_coords: {x1: 100, y1: 150, x2: 200, y2: 250}
    }
    bbox: {x1: 100, y1: 150, x2: 200, y2: 250}
    confidence: 0.95
    position: {x: 0.0, y: 0.0, z: 0.0}
  }

Response:
  success: true
  message: "Pick command accepted"
```

---

### 📥 상품 담기 요청  
- **Service**: `/pickee/arm/place_product`  
- **From → To**: Pic Main → Pic Arm

#### Request:
```plaintext
int32 robot_id
int32 order_id
int32 product_id
string arm_side                # Pickee는 "" 사용
shopee_interfaces/msg/Pose6D pose          
```

- **Pose6D**
```plaintext
float32 joint_!
float32 joint_2
float32 joint_3
float32 joint_4
float32 joint_5
float32 joint_6
```

#### Response:
```plaintext
bool success
string message
```

#### 예시:
```plaintext
Request:
  robot_id: 1
  order_id: 21
  product_id: 34
  arm_side: ""
  pose: {joint_1!: 0.0, joint_2: 0.0, joint3: 0.0, joint_4: 0.0, joint_5: 0.0, joint_6: 0.0}

Response:
  success: true
  message: "Place command accepted"
```
