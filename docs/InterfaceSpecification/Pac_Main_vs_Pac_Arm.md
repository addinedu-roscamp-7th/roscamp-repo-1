# Pac Main ↔ Pac Arm

**Main** = Shopee Main Service

**Pac Main** = Packee Main Controller

## Data Types

- **Point3D**
  - float32 x
  - float32 y
  - float32 z
- **BBox**
  - int32 x1
  - int32 y1
  - int32 x2
  - int32 y2

## Topic

### 자세 변경 상태
- **Topic:** /packee/arm/pose_status
- **From:** Pac Arm
- **To:** Pac Main

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
    pose_type: "cart_view"
    status: "in_progress"
    progress: 0.6
    message: "Moving to cart view pose"

**완료:**

    robot_id: 1
    order_id: "ORDER_001"
    pose_type: "cart_view"
    status: "completed"
    progress: 1.0
    message: "Reached cart view pose"

---

### 픽업 상태
- **Topic:** /packee/arm/pick_status
- **From:** Pac Arm
- **To:** Pac Main

#### Message
- int32 robot_id
- string order_id
- string product_id
- string arm_side
- string status
- string current_phase
- float32 progress
- string message

**arm_side 값:**
- "left" - 좌측 팔
- "right" - 우측 팔

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
**좌측 팔 - 진행 중:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    arm_side: "left"
    status: "in_progress"
    current_phase: "grasping"
    progress: 0.7
    message: "Left arm grasping product"

**우측 팔 - 완료:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_002"
    arm_side: "right"
    status: "completed"
    current_phase: "done"
    progress: 1.0
    message: "Right arm picked successfully"

**좌측 팔 - 실패:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_003"
    arm_side: "left"
    status: "failed"
    current_phase: "grasping"
    progress: 0.7
    message: "Left arm grasp failed - gripper error"

---

### 담기 상태
- **Topic:** /packee/arm/place_status
- **From:** Pac Arm
- **To:** Pac Main

#### Message
- int32 robot_id
- string order_id
- string product_id
- string arm_side
- string status
- string current_phase
- float32 progress
- string message

**arm_side 값:**
- "left" - 좌측 팔
- "right" - 우측 팔

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
**우측 팔 - 진행 중:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_002"
    arm_side: "right"
    status: "in_progress"
    current_phase: "moving"
    progress: 0.5
    message: "Right arm moving to packing box"

**좌측 팔 - 완료:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    arm_side: "left"
    status: "completed"
    current_phase: "done"
    progress: 1.0
    message: "Left arm placed successfully"

**우측 팔 - 실패:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_003"
    arm_side: "right"
    status: "failed"
    current_phase: "planning"
    progress: 0.1
    message: "Right arm path planning failed - collision detected"

---

## Service

### 자세 변경 명령
- **Service:** /packee/arm/move_to_pose
- **From:** Pac Main
- **To:** Pac Arm

#### Request
- int32 robot_id
- string order_id
- string pose_type

**pose_type 값:**
- "cart_view" - 장바구니 확인 자세
- "standby" - 대기 자세

#### Response
- bool accepted
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    pose_type: "cart_view"

**Response:**

    accepted: true
    message: "Pose change command accepted"

---

### 상품 픽업 명령
- **Service:** /packee/arm/pick_product
- **From:** Pac Main
- **To:** Pac Arm

#### Request
- int32 robot_id
- string order_id
- string product_id
- string arm_side
- Point3D target_position
- BBox bbox

**arm_side 값:**
- "left" - 좌측 팔 사용
- "right" - 우측 팔 사용

#### Response
- bool accepted
- string message

#### 예시
**Request - 좌측 팔:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    arm_side: "left"
    target_position: {x: 0.3, y: 0.15, z: 0.8}
    bbox: {x1: 120, y1: 180, x2: 250, y2: 320}

**Response:**

    accepted: true
    message: "Left arm pick command accepted"

**Request - 우측 팔:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_002"
    arm_side: "right"
    target_position: {x: 0.25, y: -0.1, z: 0.75}
    bbox: {x1: 280, y1: 150, x2: 380, y2: 280}

**Response:**

    accepted: true
    message: "Right arm pick command accepted"

---

### 상품 담기 명령
- **Service:** /packee/arm/place_product
- **From:** Pac Main
- **To:** Pac Arm

#### Request
- int32 robot_id
- string order_id
- string product_id
- string arm_side
- Point3D box_position

**arm_side 값:**
- "left" - 좌측 팔 사용
- "right" - 우측 팔 사용

#### Response
- bool accepted
- string message

#### 예시
**Request - 좌측 팔:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    arm_side: "left"
    box_position: {x: 0.5, y: 0.3, z: 0.2}

**Response:**

    accepted: true
    message: "Left arm place command accepted"

**Request - 우측 팔:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_002"
    arm_side: "right"
    box_position: {x: 0.5, y: -0.3, z: 0.2}

**Response:**

    accepted: true
    message: "Right arm place command accepted"
