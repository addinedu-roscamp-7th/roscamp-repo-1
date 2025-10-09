# Main ↔ Pac Main

**Main** = Shopee Main Service

**Pac Main** = Packee Main Controller

## Topic

### 포장 완료 알림
- **Topic:** /packee/packing_complete
- **From:** Pac Main
- **To:** Main

#### Message
- int32 robot_id
- string order_id
- bool success
- int32 packed_items
- string message

#### 예시
**성공:**

    robot_id: 1
    order_id: "ORDER_001"
    success: true
    packed_items: 5
    message: "Packing completed"

**실패:**

    robot_id: 1
    order_id: "ORDER_001"
    success: false
    packed_items: 3
    message: "Packing failed - gripper error"

---

### 로봇 상태 전송
- **Topic:** /packee/robot_status
- **From:** Pac Main
- **To:** Main

#### Message
- int32 robot_id
- string state # e.g., "packing", "idle", "error"
- string current_order_id
- int32 items_in_cart

#### 예시
    robot_id: 1
    state: "packing"
    current_order_id: "ORDER_001"
    items_in_cart: 5

---

## Service

### 작업 가능 확인 요청
- **Service:** /packee/packing/check_availability
- **From:** Main
- **To:** Pac Main
- **비고:** 실제 확인 결과는 `/packee/availability_result` 토픽으로 전송됨.

#### Request
- int32 robot_id
- string order_id

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"

**Response:**

    success: true
    message: "Availability check initiated"

---

### 포장 시작 명령
- **Service:** /packee/packing/start
- **From:** Main
- **To:** Pac Main

#### Request
- int32 robot_id
- string order_id

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"

**Response:**

    success: true
    message: "Packing started"

---

### 작업 가능 확인 완료
- **Topic:** /packee/availability_result
- **From:** Pac Main
- **To:** Main

#### Message
- int32 robot_id
- string order_id
- bool available
- bool cart_detected
- string message

#### 예시
**작업 가능:**

    robot_id: 1
    order_id: "ORDER_001"
    available: true
    cart_detected: true
    message: "Ready for packing"

**작업 불가 - 장바구니 없음:**

    robot_id: 1
    order_id: "ORDER_001"
    available: false
    cart_detected: false
    message: "Cart not detected"

**작업 불가 - 로봇 상태:**

    robot_id: 1
    order_id: "ORDER_001"
    available: false
    cart_detected: true
    message: "Robot busy with another order"
