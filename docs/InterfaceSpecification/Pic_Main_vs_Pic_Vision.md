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

## Topic 인터페이스

| 구분 | 메시지명 | 토픽 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **매대 상품 인식 완료** | `/pickee/vision/detection_result` | Topic | Pic Vision | Pic Main | `int32 robot_id`<br>`int32 order_id`<br>`bool success`<br>`DetectedProduct[] products`<br>`string message`<br><br>**DetectedProduct**<br>`int32 product_id`<br>`int32 bbox_number`<br>`BBox bbox_coords`<br>`float32 confidence`<br><br>**BBox**<br>`int32 x1, y1, x2, y2` | **성공**<br>`robot_id: 1`<br>`order_id: 4`<br>`success: true`<br>`products: [`<br>`  {`<br>`    product_id: 4`<br>`    bbox_number: 1`<br>`    bbox_coords: {x1: 100, y1: 150, x2: 200, y2: 250}`<br>`    confidence: 0.95`<br>`  },`<br>`  {`<br>`    product_id: 5`<br>`    bbox_number: 2`<br>`    bbox_coords: {x1: 250, y1: 150, x2: 350, y2: 250}`<br>`    confidence: 0.92`<br>`  }`<br>`]`<br>`message: "2 products detected"`<br><br>**실패**<br>`robot_id: 1`<br>`order_id: 4`<br>`success: false`<br>`products: []`<br>`message: "No products detected"` |
| **장바구니 내 특정 상품 확인 완료** | `/pickee/vision/cart_check_result` | Topic | Pic Vision | Pic Main | `int32 robot_id`<br>`int32 order_id`<br>`bool success`<br>`int32 product_id`<br>`bool found`<br>`int32 quantity`<br>`string message` | **상품 있음**<br>`robot_id: 1`<br>`order_id: 4`<br>`success: true`<br>`product_id: 5`<br>`found: true`<br>`quantity: 2`<br>`message: "Product found in cart"`<br><br>**상품 없음**<br>`robot_id: 1`<br>`order_id: 4`<br>`success: true`<br>`product_id: 5`<br>`found: false`<br>`quantity: 0`<br>`message: "Product not found in cart"`<br><br>**실패**<br>`robot_id: 1`<br>`order_id: 4`<br>`success: false`<br>`product_id: 5`<br>`found: false`<br>`quantity: 0`<br>`message: "Vision system error"` |
| **장애물 감지 알림** | `/pickee/vision/obstacle_detected` | Topic | Pic Vision | Pic Main | `int32 robot_id`<br>`int32 order_id`<br>`Obstacle[] obstacles`<br>`string message`<br><br>**Obstacle**<br>`string obstacle_type`<br>`Point2D position`<br>`float32 distance`<br>`float32 velocity`<br>`Vector2D direction`<br>`BBox bbox`<br>`float32 confidence`<br><br>**obstacle_type**<br>정적: `"cart"`, `"box"`, `"product"`, `"shelf"`<br>동적: `"person"`, `"other_robot"`, `"cart_moving"`<br><br>***Note**: Main은 이 정보를 Mobile에 그대로 전달하여 Mobile이 자체적으로 경로 계획 수행* | **정적 장애물**<br>`robot_id: 1`<br>`order_id: 4`<br>`obstacles: [`<br>`  {`<br>`    obstacle_type: "cart"`<br>`    position: {x: 5.2, y: 3.1}`<br>`    distance: 2.5`<br>`    velocity: 0.0`<br>`    direction: {vx: 0.0, vy: 0.0}`<br>`    bbox: {x1: 200, y1: 150, x2: 350, y2: 400}`<br>`    confidence: 0.92`<br>`  }`<br>`]`<br>`message: "1 static obstacle detected"`<br><br>**동적 장애물**<br>`robot_id: 1`<br>`order_id: 4`<br>`obstacles: [`<br>`  {`<br>`    obstacle_type: "person"`<br>`    position: {x: 8.5, y: 4.2}`<br>`    distance: 1.5`<br>`    velocity: 1.2`<br>`    direction: {vx: 0.8, vy: 0.9}`<br>`    bbox: {x1: 300, y1: 100, x2: 400, y2: 450}`<br>`    confidence: 0.96`<br>`  }`<br>`]`<br>`message: "1 dynamic obstacle detected"` |
| **추종 직원 위치** | `/pickee/vision/staff_location` | Topic | Pic Vision | Pic Main | `int32 robot_id`<br>`Point2D relative_position`<br>`float32 distance`<br>`bool is_tracking` | `robot_id: 1`<br>`relative_position: {x: 2.5, y: 0.3}`<br>`distance: 2.52`<br>`is_tracking: true` |
| **직원 등록 결과** | `/pickee/vision/register_staff_result` | Topic | Pic Vision | Pic Main | `int32 robot_id`<br>`bool success`<br>`string message` | **성공**<br>`robot_id: 1`<br>`success: true`<br>`message: "Staff registration successful."`<br><br>**실패**<br>`robot_id: 1`<br>`success: false`<br>`message: "Failed to register staff: Timed out."` |

## Service 인터페이스

| 구분 | 서비스명 | 서비스 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **매대 상품 인식 요청** | `/pickee/vision/detect_products` | Service | Pic Main | Pic Vision | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`int32[] product_ids`<br><br>**Response**<br>`bool success`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 4`<br>`product_ids: [5, 6]`<br><br>**Response**<br>`success: true`<br>`message: "Detection started"` |
| **장바구니 내 특정 상품 확인 요청** | `/pickee/vision/check_product_in_cart` | Service | Pic Main | Pic Vision | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br><br>**Response**<br>`bool success`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 4`<br>`product_id: 5`<br><br>**Response**<br>`success: true`<br>`message: "Cart product check started"` |
| **장바구니 존재 확인 요청** | `/pickee/vision/check_cart_presence` | Service | Pic Main | Pic Vision | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br><br>**Response**<br>`bool success`<br>`bool cart_present`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 4`<br><br>**Response (장바구니 있음)**<br>`success: true`<br>`cart_present: true`<br>`message: "Cart detected"`<br><br>**Response (장바구니 없음)**<br>`success: true`<br>`cart_present: false`<br>`message: "Cart not detected"` |
| **영상 송출 시작 명령** | `/pickee/vision/video_stream_start` | Service | Pic Main | Pic Vision | **Request**<br>`string user_type`<br>`string user_id`<br>`int32 robot_id`<br><br>**Response**<br>`bool success`<br>`string message` | **Request**<br>`user_type: "admin"`<br>`user_id: "admin01"`<br>`robot_id: 1`<br><br>**Response**<br>`success: true`<br>`message: "video streaming started"` |
| **영상 송출 중지 명령** | `/pickee/vision/video_stream_stop` | Service | Pic Main | Pic Vision | **Request**<br>`string user_type`<br>`string user_id`<br>`int32 robot_id`<br><br>**Response**<br>`bool success`<br>`string message` | **Request**<br>`user_type: "admin"`<br>`user_id: "admin01"`<br>`robot_id: 1`<br><br>**Response**<br>`success: true`<br>`message: "video streaming stopped"` |
| **직원 등록 요청** | `/pickee/vision/register_staff` | Service | Pic Main | Pic Vision | **Request**<br>`int32 robot_id`<br><br>**Response**<br>`bool accepted`<br>`string message` | **Request**<br>`robot_id: 1`<br><br>**Response**<br>`accepted: true`<br>`message: "Staff registration process accepted."` |
| **직원 추종 제어** | `/pickee/vision/track_staff` | Service | Pic Main | Pic Vision | **Request**<br>`int32 robot_id`<br>`bool track`<br><br>**Response**<br>`bool success`<br>`string message`<br><br>**track**<br>`true` - 추종 시작<br>`false` - 추종 중지 | **Request (추종 시작)**<br>`robot_id: 1`<br>`track: true`<br><br>**Response**<br>`success: true`<br>`message: "Started tracking STAFF_001"`<br><br>**Request (추종 중지)**<br>`robot_id: 1`<br>`track: false`<br><br>**Response**<br>`success: true`<br>`message: "Stopped tracking STAFF_001"` |
| **Vision 모드 설정** | `/pickee/vision/set_mode` | Service | Pic Main | Pic Vision | **Request**<br>`int32 robot_id`<br>`string mode`<br><br>**Response**<br>`bool success`<br>`string message`<br><br>**mode**<br>`"navigation"`, `"register_staff"`, `"detect_products"`, `"track_staff"` | **Request**<br>`robot_id: 1`<br>`mode: "register_staff"`<br><br>**Response**<br>`success: true`<br>`message: "Vision mode switched to register_staff"` |
| **음성 송출 요청** | `/pickee/tts_request` | Service | Pic Vision | Pic Main | **Request**<br>`string text_to_speak`<br><br>**Response**<br>`bool success`<br>`string message` | **Request**<br>`text_to_speak: "뒤로 돌아주세요."`<br><br>**Response**<br>`success: true`<br>`message: "TTS completed."` |