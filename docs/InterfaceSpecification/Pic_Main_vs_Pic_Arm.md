# Pic Main ↔ Pic Arm

**Pic Main** = Pickee Main Controller

**Pic Arm** = Pickee Arm Controller

## Topic

### 자세 변경 상태
- **Topic:** /pickee/arm/pose_status
- **From:** Pic Arm
- **To:** Pic Main

#### Message
- int32 robot_id
- string order_id
- string pose_type
- string status
- float32 progress
- string message

**status 값:**
- "in_progress" - 진행 중
- "completed" - 완료
- "failed" - 실패

#### 예시
**진행 중:**

    robot_id: 1
    order_id: "ORDER_001"
    pose_type: "shelf_view"
    status: "in_progress"
    progress: 0.6
    message: "Moving to shelf view pose"

**완료:**

    robot_id: 1
    order_id: "ORDER_001"
    pose_type: "shelf_view"
    status: "completed"
    progress: 1.0
    message: "Reached shelf view pose"

**실패:**

    robot_id: 1
    order_id: "ORDER_001"
    pose_type: "cart_view"
    status: "failed"
    progress: 0.3
    message: "Joint limit exceeded"

---

### 픽업 상태
- **Topic:** /pickee/arm/pick_status
- **From:** Pic Arm
- **To:** Pic Main

#### Message
- int32 robot_id
- string order_id
- string product_id
- string status
- string current_phase
- float32 progress
- string message

**status 값:**
- "in_progress" - 진행 중
- "completed" - 완료
- "failed" - 실패

**current_phase 값:**
- "planning" - 경로 계획 중
- "approaching" - 접근 중
- "grasping" - 그립 중
- "lifting" - 들어올리는 중
- "done" - 완료

#### 예시
**진행 중 - 경로 계획:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    status: "in_progress"
    current_phase: "planning"
    progress: 0.2
    message: "Planning grasp trajectory"

**진행 중 - 그립:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    status: "in_progress"
    current_phase: "grasping"
    progress: 0.7
    message: "Grasping product"

**완료:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    status: "completed"
    current_phase: "done"
    progress: 1.0
    message: "Product picked successfully"

**실패:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_002"
    status: "failed"
    current_phase: "grasping"
    progress: 0.7
    message: "Grasp failed - gripper error"

---

### 담기 상태
- **Topic:** /pickee/arm/place_status
- **From:** Pic Arm
- **To:** Pic Main

#### Message
- int32 robot_id
- string order_id
- string product_id
- string status
- string current_phase
- float32 progress
- string message

**status 값:**
- "in_progress" - 진행 중
- "completed" - 완료
- "failed" - 실패

**current_phase 값:**
- "planning" - 경로 계획 중
- "moving" - 이동 중
- "placing" - 놓는 중
- "releasing" - 그립 해제 중
- "done" - 완료

#### 예시
**진행 중 - 이동:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    status: "in_progress"
    current_phase: "moving"
    progress: 0.5
    message: "Moving to cart"

**완료:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    status: "completed"
    current_phase: "done"
    progress: 1.0
    message: "Product placed in cart successfully"

**실패:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_002"
    status: "failed"
    current_phase: "planning"
    progress: 0.1
    message: "Path planning failed - collision detected"

---

## Service

### 자세 변경 요청
- **Service:** /pickee/arm/move_to_pose
- **From:** Pic Main
- **To:** Pic Arm

#### Request
- int32 robot_id
- string order_id
- string pose_type

**pose_type 값:**
- "shelf_view" - 매대 확인 자세
- "cart_view" - 장바구니 확인 자세
- "standby" - 대기 자세

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    pose_type: "shelf_view"

**Response:**

    success: true
    message: "Pose change command accepted"

---

### 상품 픽업 요청
- **Service:** /pickee/arm/pick_product
- **From:** Pic Main
- **To:** Pic Arm

#### Request
- int32 robot_id
- string order_id
- string product_id
- Point3D target_position

#### Response
- bool accepted
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    target_position: {x: 0.5, y: 0.3, z: 1.2}

**Response:**

    accepted: true
    message: "Pick command accepted"

---

### 상품 담기 요청
- **Service:** /pickee/arm/place_product
- **From:** Pic Main
- **To:** Pic Arm

#### Request
- int32 robot_id
- string order_id
- string product_id

#### Response
- bool accepted
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"

**Response:**

    accepted: true
    message: "Place command accepted"
