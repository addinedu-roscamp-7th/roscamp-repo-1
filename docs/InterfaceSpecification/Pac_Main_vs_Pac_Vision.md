Pac Main = Packee Main Controller

Pac Vision = Packee Vision AI Service

## ROS 인터페이스 매핑

| 분류 | 토픽/서비스 | ROS 타입 | 파일 경로 |
|---|---|---|---|
| Service | `/packee/vision/check_cart_presence` | `shopee_interfaces/srv/PackeeVisionCheckCartPresence` | `ros2_ws/src/shopee_interfaces/srv/PackeeVisionCheckCartPresence.srv` |
| Service | `/packee/vision/detect_products_in_cart` | `shopee_interfaces/srv/PackeeVisionDetectProductsInCart` | `ros2_ws/src/shopee_interfaces/srv/PackeeVisionDetectProductsInCart.srv` |
| Service | `/packee/vision/verify_packing_complete` | `shopee_interfaces/srv/PackeeVisionVerifyPackingComplete` | `ros2_ws/src/shopee_interfaces/srv/PackeeVisionVerifyPackingComplete.srv` |

**구조체 매핑**
- `DetectedProduct` → `shopee_interfaces/msg/PackeeDetectedProduct`
- `BBox` → `shopee_interfaces/msg/BBox`
- `Point3D` → `shopee_interfaces/msg/Point3D`




From

To

Message

예시

Service











장바구니 유무 확인

/packee/vision/check_cart_presence

Pac Main

Pac Vision

# Request
int32 robot_id
---
# Response
bool cart_present
float32 confidence
string message

# Request
robot_id: 1

# Response - 장바구니 있음
cart_present: true
confidence: 0.98
message: "Cart detected"

# Response - 장바구니 없음
cart_present: false
confidence: 0.95
message: "No cart detected"

장바구니 내 상품 위치 확인

/packee/vision/detect_products_in_cart

Pac Main

Pac Vision

# Request
int32 robot_id
int32 order_id
int32[] expected_product_ids

# Response
bool success
DetectedProduct[] products
int32 total_detected
string message

# DetectedProduct
int32 product_id
BBox bbox
float32 confidence
Point3D position

# BBox
int32 x1
int32 y1
int32 x2
int32 y2

# Point3D
float32 x
float32 y
float32 z

# Request
robot_id: 1
order_id: 3
expected_product_ids: [1, 2, 3]

# Response - 성공
success: true
products:
  - product_id: 3
    bbox: {x1: 120, y1: 180, x2: 250, y2: 320}
    confidence: 0.94
    position: {x: 0.3, y: 0.15, z: 0.8}
  - product_id: 4
    bbox: {x1: 280, y1: 150, x2: 380, y2: 280}
    confidence: 0.91
    position: {x: 0.25, y: -0.1, z: 0.75}
  - product_id: 5
    bbox: {x1: 400, y1: 200, x2: 520, y2: 340}
    confidence: 0.89
    position: {x: 0.2, y: 0.2, z: 0.7}
total_detected: 3
message: "All products detected"

# Response - 일부만 감지
success: true
products:
  - product_id: 3
    bbox: {x1: 120, y1: 180, x2: 250, y2: 320}
    confidence: 0.94
    position: {x: 0.3, y: 0.15, z: 0.8}
  - product_id: 2
    bbox: {x1: 280, y1: 150, x2: 380, y2: 280}
    confidence: 0.91
    position: {x: 0.25, y: -0.1, z: 0.75}
total_detected: 2
message: "Detected 2 out of 3 products"

# Response - 실패
success: false
products: []
total_detected: 0
message: "No products detected in cart"

포장 완료 확인

/packee/vision/verify_packing_complete

Pac Main

Pac Vision

# Request
int32 robot_id
int32 order_id
---
# Response
bool cart_empty
int32 remaining_items
int32[] remaining_product_ids
string message

# Request
robot_id: 1
order_id: 3

# Response - 포장 완료
cart_empty: true
remaining_items: 0
remaining_product_ids: []
message: "Cart is empty, packing complete"

# Response - 일부 남음
cart_empty: false
remaining_items: 1
remaining_product_ids: [3]
message: "1 item(s) remaining in cart"
