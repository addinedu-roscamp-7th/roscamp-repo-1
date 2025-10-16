Pic Main = Pickee Main Controller

Pic Vision = Pickee Vision AI Service

### `/pickee/vision/detection_result`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeVisionDetection.msg`

### `/pickee/vision/cart_check_result`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeVisionCartCheck.msg`

### `/pickee/vision/obstacle_detected`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeVisionObstacles.msg`

### `/pickee/vision/staff_location`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeVisionStaffLocation.msg`

### `/pickee/vision/register_staff_result`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeVisionStaffRegister.msg`

### `/pickee/vision/detect_products`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionDetectProducts.srv`

### `/pickee/vision/check_product_in_cart`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionCheckProductInCart.srv`

### `/pickee/vision/check_cart_presence`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionCheckCartPresence.srv`

### `/pickee/vision/video_stream_start`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionVideoStreamStart.srv`

### `/pickee/vision/video_stream_stop`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionVideoStreamStop.srv`

### `/pickee/vision/register_staff`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionRegisterStaff.srv`

### `/pickee/vision/track_staff`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionTrackStaff.srv`

### `/pickee/vision/set_mode`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionSetMode.srv`

### `/pickee/tts_request`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeTtsRequest.srv`

**구조체 매핑**
- `DetectedProduct` → `shopee_interfaces/msg/PickeeDetectedProduct`
- `Obstacle` → `shopee_interfaces/msg/Obstacle`
- `BBox` → `shopee_interfaces/msg/BBox`
- `Point2D` → `shopee_interfaces/msg/Point2D`



From

To

Message

예시

Topic











매대 상품 인식 완료



/pickee/vision/detection_result

Pic Vision

Pic Main

int32 robot_id
int32 order_id
bool success
DetectedProduct[] products
string message

# DetectedProduct
int32 product_id
int32 bbox_number
BBox bbox_coords
float32 confidence

# BBox
int32 x1
int32 y1
int32 x2
int32 y2

# 성공
robot_id: 1
order_id: 4
success: true
products: [
  {
    product_id: 4
    bbox_number: 1
    bbox_coords: {x1: 100, y1: 150, x2: 200, y2: 250}
    confidence: 0.95
  },
  {
    product_id: 5
    bbox_number: 2
    bbox_coords: {x1: 250, y1: 150, x2: 350, y2: 250}
    confidence: 0.92
  }
]
message: "2 products detected"

# 실패
robot_id: 1
order_id: 4
success: false
products: []
message: "No products detected"

장바구니 내 특정 상품 확인 완료

/pickee/vision/cart_check_result

Pic Vision

Pic Main

int32 robot_id
int32 order_id
bool success
int32 product_id
bool found
int32 quantity
string message

# 상품 있음
robot_id: 1
order_id: 4
success: true
product_id: 5
found: true
quantity: 2
message: "Product found in cart"

# 상품 없음
robot_id: 1
order_id: 4
success: true
product_id: 5
found: false
quantity: 0
message: "Product not found in cart"

# 실패
robot_id: 1
order_id: 4
success: false
product_id: 5
found: false
quantity: 0
message: "Vision system error"

장애물 감지 알림

/pickee/vision/obstacle_detected

Pic Vision

Pic Main

int32 robot_id
int32 order_id
Obstacle[] obstacles
string message

# Obstacle
string obstacle_type  # 정적: "cart", "box", "product", "shelf"
                      # 동적: "person", "other_robot", "cart_moving"
Point2D position      # 장애물 중심 위치 (m)
float32 distance      # 로봇으로부터의 거리 (m)
float32 velocity      # 속도 (m/s) - 정적: 0.0, 동적: > 0.0
Vector2D direction    # 이동 방향 (동적 장애물만)
BBox bbox            # Bounding Box
float32 confidence   # 인식 신뢰도

# Point2D
float32 x
float32 y

# Vector2D
float32 vx  # x 방향 속도 성분
float32 vy  # y 방향 속도 성분

# BBox (재사용)
int32 x1
int32 y1
int32 x2
int32 y2

# 정적 장애물
robot_id: 1
order_id: 4
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

# 동적 장애물
robot_id: 1
order_id: 4
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

# 복합 장애물 (정적 + 동적)
robot_id: 1
order_id: 4
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

추종 직원 위치

/pickee/vision/staff_location

Pic Vision

Pic Main

int32 robot_id
Point2D relative_position (로봇 기준 직원의 상대 위치)
float32 distance
bool is_tracking

robot_id: 1
relative_position: {x: 2.5, y: 0.3}
distance: 2.52
is_tracking: true

직원 등록 결과

/pickee/vision/register_staff_result

Pic Vision

Pic Main

int32 robot_id
bool success
string message

# 성공
robot_id: 1
success: true
message: "Staff registration successful."

# 실패
robot_id: 1
success: false
message: "Failed to register staff: Timed out."

Service











매대 상품 인식 요청



/pickee/vision/detect_products

Pic Main

Pic Vision

# Request
int32 robot_id
int32 order_id
int32[] product_ids

---
# Response
bool success
string message

# Request
robot_id: 1
order_id: 4
product_ids: [5, 6]

# Response
success: true
message: "Detection started"

장바구니 내 특정 상품 확인 요청

/pickee/vision/check_product_in_cart

Pic Main

Pic Vision

# Request
int32 robot_id
int32 order_id
int32 product_id

---
# Response
bool success
string message

# Request
robot_id: 1
order_id: 4
product_id: 5

# Response
success: true
message: "Cart product check started"

장바구니 존재 확인 요청

/pickee/vision/check_cart_presence

Pic Main

Pic Vision

# Request
int32 robot_id
int32 order_id

---
# Response
bool success
bool cart_present
string message

# Request
robot_id: 1
order_id: 4

# Response (장바구니 있음)
success: true
cart_present: true
message: "Cart detected"

# Response (장바구니 없음)
success: true
cart_present: false
message: "Cart not detected"

영상 송출 시작 명령

/pickee/vision/video_stream_start

Pic Main

Pic Vision

# Request
string user_type
string user_id
int32 robot_id

---
# Response
bool success
string message

# Request
user_type: "admin"
user_id: "admin01"
robot_id: 1

# Response
success: true
message: "video streaming started"

영상 송출 중지 명령

/pickee/vision/video_stream_stop

Pic Main

Pic Vision

# Request
string user_type
string user_id
int32 robot_id

---
# Response
bool success
string message

# Request
user_type: "admin"
user_id: "admin01"
robot_id: 1

# Response
success: true
message: "video streaming stopped"

직원 등록 요청

/pickee/vision/register_staff

Pic Main

Pic Vision

# Request
int32 robot_id
---
# Response
bool accepted (작업 접수 여부)
string message

# Request
robot_id: 1
---
# Response
accepted: true
message: "Staff registration process accepted."

직원 추종 제어

/pickee/vision/track_staff

Pic Main

Pic Vision

# Request
int32 robot_id
bool track (true: 추종 시작, false: 추종 중지)
---
# Response
bool success
string message

# Request
robot_id: 1
track: true
Response:
---
# Response
success: true
message: "Started tracking STAFF_001"

# Request
robot_id: 1
track: false
---
# Response
success: true
message: "Stopped tracking STAFF_001"

Vision 모드 설정

/pickee/vision/set_mode

Pic Main

Pic Vision

# Request
int32 robot_id
string mode ("navigation", "register_staff", "detect_products", "track_staff")
---
# Response
bool success
string message

# Request
robot_id: 1
mode: "register_staff"
---
# Response
success: true
message: "Vision mode switched to register_staff"

음성 송출 요청

/pickee/tts_request

Pic Vision

Pic Main

# Request
string text_to_speak
---
# Response
bool success
string message

# Request
text_to_speak: "뒤로 돌아주세요."
---
# Response
success: true
message: "TTS completed."
