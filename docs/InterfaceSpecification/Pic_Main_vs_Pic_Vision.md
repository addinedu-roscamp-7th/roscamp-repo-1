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