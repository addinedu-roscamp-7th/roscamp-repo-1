# Main ↔ Pic Main

**Main** = Shopee Main Service

**Pic Main** = Pickee Main Controller

## Topic

### 이동 시작 알림
- **Topic:** /pickee/moving_status
- **From:** Pic Main
- **To:** Main

#### Message
- int32 robot_id
- string order_id
- string location_id

#### 예시
    robot_id: 1
    order_id: "ORDER_001"
    location_id: "LOC_A1"

---

### 도착 보고
- **Topic:** /pickee/arrival_notice
- **From:** Pic Main
- **To:** Main

#### Message
- int32 robot_id
- string order_id
- string location_id
- string shelf_id

#### 예시
    robot_id: 1
    order_id: "ORDER_001"
    location_id: "LOC_A1"
    shelf_id: "SHELF_A1_01"

---

### 상품 위치 인식 완료
- **Topic:** /pickee/product_detected
- **From:** Pic Main
- **To:** Main

#### Message
- int32 robot_id
- string order_id
- DetectedProduct[] products

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
    robot_id: 1
    order_id: "ORDER_001"
    products: [
      {
        product_id: "PROD_001",
        bbox_number: 1,
        bbox_coords: {x1: 100, y1: 150, x2: 200, y2: 250},
        confidence: 0.95
      },
      {
        product_id: "PROD_002",
        bbox_number: 2,
        bbox_coords: {x1: 250, y1: 150, x2: 350, y2: 250},
        confidence: 0.92
      }
    ]

---

### 장바구니 교체 완료
- **Topic:** /pickee/cart_handover_complete
- **From:** Pic Main
- **To:** Main

#### Message
- int32 robot_id
- string order_id

#### 예시
    robot_id: 1
    order_id: "ORDER_001"

---

### 담기 완료 보고
- **Topic:** /pickee/product/selection_result
- **From:** Pic Main
- **To:** Main
- **발행:** 담기 완료 시 1회

#### Message
- int32 robot_id
- string order_id
- string product_id
- bool success
- int32 quantity
- string message

#### 예시
**성공:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    success: true
    quantity: 1
    message: "Pick completed"

**실패:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    success: false
    quantity: 0
    message: "Pick failed - gripper error"

---

### 로봇 상태 전송
- **Topic:** /pickee/robot_status
- **From:** Pic Main
- **To:** Main

#### Message
- int32 robot_id
- string state # e.g., "moving", "picking", "idle", "error"
- float32 battery_level
- string current_order_id
- float32 position_x
- float32 position_y
- float32 orientation_z

#### 예시
    robot_id: 1
    state: "shopping"
    battery_level: 75.5
    current_order_id: "ORDER_001"
    position_x: 10.5
    position_y: 5.2
    orientation_z: 3.14

---

## Service

### 작업 시작 명령
- **Service:** /pickee/workflow/start_task
- **From:** Main
- **To:** Pic Main

#### Request
- int32 robot_id
- string order_id
- string customer_id
- ProductLocation[] product_list

**ProductLocation 구조:**
- string product_id
- string location_id
- string shelf_id
- int32 quantity

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    customer_id: "customer001"
    product_list: [
      {
        product_id: "PROD_001",
        location_id: "LOC_A1",
        shelf_id: "SHELF_A1_01",
        quantity: 2
      }
    ]

**Response:**

    success: true
    message: "Task started"

---

### 매대 이동 명령
- **Service:** /pickee/workflow/move_to_shelf
- **From:** Main
- **To:** Pic Main

#### Request
- int32 robot_id
- string order_id
- string location_id
- string shelf_id

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    location_id: "LOC_A1"
    shelf_id: "SHELF_A1_01"

**Response:**

    success: true
    message: "Moving to shelf"

---

### 상품 인식 명령
- **Service:** /pickee/product/detect
- **From:** Main
- **To:** Pic Main

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
    order_id: "ORDER_001",
    product_ids: ["PROD_001", "PROD_002"]

**Response:**

    success: true
    message: "Detecting products"

---

### 상품 처리 명령
- **Service:** /pickee/product/process_selection
- **From:** Main
- **To:** Pic Main

#### Request
- int32 robot_id
- string order_id
- string product_id
- int32 bbox_number
- int32 quantity

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    product_id: "PROD_001"
    bbox_number: 1
    quantity: 1

**Response:**

    success: true
    message: "Processing selection started"

---

### 쇼핑 종료 명령
- **Service:** /pickee/workflow/end_shopping
- **From:** Main
- **To:** Pic Main

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
    message: "Shopping ended"

---

### 포장대 이동 명령
- **Service:** /pickee/workflow/move_to_packaging
- **From:** Main
- **To:** Pic Main

#### Request
- int32 robot_id
- string order_id
- string location_id

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    location_id: "PACKAGING_A"

**Response:**

    success: true
    message: "Moving to packaging area"

---

### 복귀 명령
- **Service:** /pickee/workflow/return_to_base
- **From:** Main
- **To:** Pic Main

#### Request
- int32 robot_id
- string destination

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    destination: "waiting_area"

**Response:**

    success: true
    message: "Returning to base"

---

### 영상 송출 시작 명령
- **Service:** /pickee/video_stream/start
- **From:** Main
- **To:** Pic Main

#### Request
- string user_type
- string customer_id
- int32 robot_id

#### Response
- bool success
- string message

#### 예시
**Request:**

    user_type: "admin",
    customer_id: "admin01",
    robot_id: 1

**Response:**

    success: true
    message: "video streaming started"

---

### 영상 송출 중지 명령
- **Service:** /pickee/video_stream/stop
- **From:** Main
- **To:** Pic Main

#### Request
- string user_type
- string customer_id
- int32 robot_id

#### Response
- bool success
- string message

#### 예시
**Request:**

    user_type: "admin",
    customer_id: "admin01",
    robot_id: 1

**Response:**

    success: true
    message: "video streaming stopped"

---

### 상품 위치 조회
- **Service:** /main/get_product_location
- **From:** Pic Main
- **To:** Main

#### Request
- string product_id

#### Response
- bool success
- string warehouse_location
- string shelf_location
- string message

#### 예시
**Request:**

    product_id: "PROD_005"

**Response:**

    success: true
    warehouse_location: "LOC_B2"
    shelf_location: "SHELF_B2_03",
    message: "Data retrieved successfully"

---