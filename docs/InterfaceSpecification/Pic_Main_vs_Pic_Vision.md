# Pic Main ↔ Pic Vision

**Pic Main** = Pickee Main Controller

**Pic Vision** = Pickee Vision AI Service

## Topic

### 매대 상품 인식 완료
- **Topic:** /pickee/vision/detection_result
- **From:** Pic Vision
- **To:** Pic Main

#### Message
- int32 robot_id
- string order_id
- bool success
- DetectedProduct[] products
- string message

**DetectedProduct 구조:**
- string product_id
- int32 bbox_number
- BBox bbox_coords
- float32 confidence

**BBox 구조:**
- int32 x1
- int32 y1
- int32 x2
- int32 y2

#### 예시
**성공:**

    robot_id: 1
    order_id: "ORDER_001"
    success: true
    products: [
      {
        product_id: "PROD_001"
        bbox_number: 1
        bbox_coords: {x1: 100, y1: 150, x2: 200, y2: 250}
        confidence: 0.95
      },
      {
        product_id: "PROD_002"
        bbox_number: 2
        bbox_coords: {x1: 250, y1: 150, x2: 350, y2: 250}
        confidence: 0.92
      }
    ]
    message: "2 products detected"

**실패:**

    robot_id: 1
    order_id: "ORDER_001"
    success: false
    products: []
    message: "No products detected"

---

### 장바구니 내 특정 상품 확인 완료
- **Topic:** /pickee/vision/cart_check_result
- **From:** Pic Vision
- **To:** Pic Main

#### Message
- int32 robot_id
- string order_id
- bool success
- string product_id
- bool found
- int32 quantity
- string message

#### 예시
**상품 있음:**

    robot_id: 1
    order_id: "ORDER_001"
    success: true
    product_id: "PROD_001"
    found: true
    quantity: 2
    message: "Product found in cart"

**상품 없음:**

    robot_id: 1
    order_id: "ORDER_001"
    success: true
    product_id: "PROD_001"
    found: false
    quantity: 0
    message: "Product not found in cart"

**실패:**

    robot_id: 1
    order_id: "ORDER_001"
    success: false
    product_id: "PROD_001"
    found: false
    quantity: 0
    message: "Vision system error"

---

### 장애물 감지 알림
- **Topic:** /pickee/vision/obstacle_detected
- **From:** Pic Vision
- **To:** Pic Main

#### Message
- int32 robot_id
- string order_id
- Obstacle[] obstacles
- string message

**Obstacle 구조:**
- string obstacle_type (정적: "cart", "box", "product", "shelf" / 동적: "person", "other_robot", "cart_moving")
- Point2D position (장애물 중심 위치, m)
- float32 distance (로봇으로부터의 거리, m)
- float32 velocity (속도 m/s - 정적: 0.0, 동적: > 0.0)
- Vector2D direction (이동 방향, 동적 장애물만)
- BBox bbox (Bounding Box)
- float32 confidence (인식 신뢰도)

**Point2D 구조:**
- float32 x
- float32 y

**Vector2D 구조:**
- float32 vx (x 방향 속도 성분)
- float32 vy (y 방향 속도 성분)

**BBox 구조:**
- int32 x1
- int32 y1
- int32 x2
- int32 y2

#### 예시
**정적 장애물:**

    robot_id: 1
    order_id: "ORDER_001"
    obstacles: [
      {
        obstacle_type: "cart"
        position: {x: 5.2, y: 3.1}
        distance: 2.5
        velocity: 0.0
        direction: {vx: 0.0, vy: 0.0}
        bbox: {x1: 200, y1: 150, x2: 350, y2: 400}
        confidence: 0.92
      }
    ]
    message: "1 static obstacle detected"

**동적 장애물:**

    robot_id: 1
    order_id: "ORDER_001"
    obstacles: [
      {
        obstacle_type: "person"
        position: {x: 8.5, y: 4.2}
        distance: 1.5
        velocity: 1.2
        direction: {vx: 0.8, vy: 0.9}
        bbox: {x1: 300, y1: 100, x2: 400, y2: 450}
        confidence: 0.96
      }
    ]
    message: "1 dynamic obstacle detected"

**복합 장애물 (정적 + 동적):**

    robot_id: 1
    order_id: "ORDER_001"
    obstacles: [
      {
        obstacle_type: "cart"
        distance: 3.0
        velocity: 0.0
      },
      {
        obstacle_type: "person"
        distance: 1.8
        velocity: 1.0
      }
    ]
    message: "1 static, 1 dynamic obstacle detected"

---

### 추종 직원 위치
- **Topic:** /pickee/vision/staff_location
- **From:** Pic Vision
- **To:** Pic Main

#### Message
- int32 robot_id
- Point2D relative_position (로봇 기준 직원의 상대 위치)
- float32 distance
- bool is_tracking

#### 예시
    robot_id: 1
    relative_position: {x: 2.5, y: 0.3}
    distance: 2.52
    is_tracking: true

---

### 직원 등록 결과
- **Topic:** /pickee/vision/register_staff_result
- **From:** Pic Vision
- **To:** Pic Main

#### Message
- int32 robot_id
- bool success
- string message

#### 예시
(추가 예정)

---

## Service

### 매대 상품 인식 요청
- **Service:** /pickee/vision/detect_products
- **From:** Pic Main
- **To:** Pic Vision

#### Request
- int32 robot_id
- string order_id
- string[] product_ids

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    product_ids: ["PROD_001", "PROD_002"]

**Response:**

    success: true
    message: "Detection started"

---

### 장바구니 내 특정 상품 확인 요청
- **Service:** /pickee/vision/check_product_in_cart
- **From:** Pic Main
- **To:** Pic Vision

#### Request
- int32 robot_id
- string order_id
- string product_id

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"

**Response:**

    success: true
    message: "Cart product check started"

---

### 장바구니 존재 확인 요청
- **Service:** /pickee/vision/check_cart_presence
- **From:** Pic Main
- **To:** Pic Vision

#### Request
- int32 robot_id
- string order_id

#### Response
- bool success
- bool cart_present
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"

**Response (장바구니 있음):**

    success: true
    cart_present: true
    message: "Cart detected"

**Response (장바구니 없음):**

    success: true
    cart_present: false
    message: "Cart not detected"

---

### 직원 등록 요청 (비동기 시작)
- **Service:** /pickee/vision/register_staff
- **Description:** 직원의 정면 및 후면 특징을 모두 등록하는 긴 작업을 시작하도록 요청한다. 서비스는 즉시 응답하며, 최종 결과는 `/pickee/vision/register_staff_result` 토픽으로 전송된다.
- **From:** Pic Main
- **To:** Pic Vision

#### Request
- int32 robot_id

#### Response
- bool accepted (작업 접수 여부)
- string message

#### 예시
**Request:**

    robot_id: 1

**Response:**

    accepted: true
    message: "Staff registration process accepted."

---

### 직원 추종 제어
- **Service:** /pickee/vision/track_staff
- **From:** Pic Main
- **To:** Pic Vision

#### Request
- int32 robot_id
- bool track (true: 추종 시작, false: 추종 중지)

#### Response
- bool success
- string message

#### 예시
**Request (추종 시작):**

    robot_id: 1
    track: true

**Response:**

    success: true
    message: "Started tracking STAFF_001"

**Request (추종 중지):**

    robot_id: 1
    track: false

**Response:**

    success: true
    message: "Stopped tracking STAFF_001"

---

### Vision 모드 설정
- **Service:** /pickee/vision/set_mode
- **From:** Pic Main
- **To:** Pic Vision

#### Request
- int32 robot_id
- string mode ("navigation", "register_staff", "detect_products", "track_staff")

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    mode: "register_staff"

**Response:**

    success: true
    message: "Vision mode switched to register_staff"

---

### 음성 송출 요청
- **Service:** /pickee/tts_request
- **Description:** Vision 모듈이 Main Controller에게 특정 문장의 음성 송출(TTS)을 요청한다. Main Controller는 음성 송출이 완료된 후 응답한다.
- **From:** Pic Vision
- **To:** Pic Main

#### Request
- string text_to_speak

#### Response
- bool success
- string message

#### 예시
**Request:**

    text_to_speak: "뒤로 돌아주세요."

**Response:**

    success: true
    message: "TTS completed."