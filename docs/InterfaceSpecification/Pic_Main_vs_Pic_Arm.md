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
string status          # "in_progress", "completed", "failed"
string current_phase   # "planning", "moving", "placing", "releasing", "done"
float32 progress
string message
```

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
PickeeDetectedProduct target_product
```

- **PickeeDetectedProduct**
```plaintext
int32 product_id
int32 bbox_number
DetectionInfo detection_info
float32 confidence
```

- **DetectionInfo**
```plaintext
Point2D[] polygon     # 다각형 꼭짓점 좌표 리스트
BBox bbox_coords
```

- **Point2D**
```plaintext
float32 x
float32 y
```

- **BBox**
```plaintext
int32 x1
int32 y1
int32 x2
int32 y2
```

#### Response:
```plaintext
bool accepted
string message
```

#### 예시:
```plaintext
Request:
robot_id: 1
order_id: 3
target_product: {
  product_id: 4
  bbox_number: 1
  detection_info: {
    polygon: [...]
    bbox_coords: {x1: 100, y1: 150, x2: 200, y2: 250}
  }
  confidence: 0.95
}

Response:
accepted: true
message: "Pick command accepted"
```

📝 *2025.10.20 - DetectionInfo 사용으로 polygon 정보 포함*

---

### 📥 상품 담기 요청  
- **Service**: `/pickee/arm/place_product`  
- **From → To**: Pic Main → Pic Arm

#### Request:
```plaintext
int32 robot_id
int32 order_id
int32 product_id
```

#### Response:
```plaintext
bool accepted
string message
```

#### 예시:
```plaintext
Request:
robot_id: 1
order_id: 21
product_id: 34

Response:
accepted: true
message: "Place command accepted"
```
