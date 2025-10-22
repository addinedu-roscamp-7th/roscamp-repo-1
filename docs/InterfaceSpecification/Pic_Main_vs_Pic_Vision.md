Pic Main = Pickee Main Controller

Pic Vision = Pickee Vision AI Service

---

## ROS2 Topic

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

---

## ROS2 Service

### `/pickee/vision/detect_products`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionDetectProducts.srv`

### `/pickee/vision/check_product_in_cart`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeVisionCheckProductInCart.srv`

### `/pickee/vision/check_cart_presence`
> **ROS2 Interface:** `shopee_interfaces/srv/VisionCheckCartPresence.srv`

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
- `DetectedProduct` → `shopee_interfaces/msg/DetectedProduct` (Pickee/Packee 공통)
- `Obstacle` → `shopee_interfaces/msg/Obstacle`
- `BBox` → `shopee_interfaces/msg/BBox`
- `Point2D` → `shopee_interfaces/msg/Point2D`
- `DetectionInfo` → `shopee_interfaces/msg/DetectionInfo`

**DetectedProduct 필드 사용 규칙 (Pickee)**
- 사용 필드: `product_id`, `confidence`, `bbox`, `bbox_number`, `detection_info`
- 미사용 필드: `position` (0, 0, 0)

## 👁️‍🗨️ 인터페이스 상세 정의

## 📦 메시지 (Messages)

---

### 🧾 매대 상품 인식 완료
- **Topic**: `/pickee/vision/detection_result`  
- **From → To**: Pic Vision → Pic Main  
- **Message Fields**:
```plaintext
int32 robot_id
int32 order_id
bool success
shopee_interfaces/msg/DetectedProduct[] products
string message
```

- **DetectedProduct** (Pickee 사용 필드)
```plaintext
int32 product_id
float32 confidence
shopee_interfaces/msg/BBox bbox
int32 bbox_number         # 앱 UI 선택용
shopee_interfaces/msg/DetectionInfo detection_info
shopee_interfaces/msg/Pose6D pose         
```

- **DetectionInfo**
```plaintext
shopee_interfaces/msg/Point2D[] polygon       # 다각형 꼭짓점 좌표 리스트
shopee_interfaces/msg/BBox bbox_coords
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

📝 *2025.10.20 - DetectionInfo 사용, BBox 대체*

---

### 🧺 장바구니 내 특정 상품 확인 완료
- **Topic**: `/pickee/vision/cart_check_result`  
- **From → To**: Pic Vision → Pic Main  
- **Message Fields**:
```plaintext
int32 robot_id
int32 order_id
bool success
int32 product_id
bool found
int32 quantity
string message
```

---

### ⚠️ 장애물 감지 알림
- **Topic**: `/pickee/vision/obstacle_detected`  
- **From → To**: Pic Vision → Pic Main  
- **Message Fields**:
```plaintext
int32 robot_id
int32 order_id
shopee_interfaces/msg/Obstacle[] obstacles
string message
```

- **Obstacle**
```plaintext
string obstacle_type    # 예: "cart", "box", "product", "shelf", "person", "other_robot", "cart_moving"
shopee_interfaces/msg/Point2D position        # 장애물 중심 위치 (m)
float32 distance        # 로봇으로부터의 거리 (m)
float32 velocity        # 속도 (m/s)
shopee_interfaces/msg/Vector2D direction      # 동적 장애물만 해당
shopee_interfaces/msg/BBox bbox
float32 confidence      # 인식 신뢰도
```

- **Point2D**
```plaintext
float32 x
float32 y
```

- **Vector2D**
```plaintext
float32 vx
float32 vy
```

- **BBox**
```plaintext
int32 x1
int32 y1
int32 x2
int32 y2
```

---

### 🧍 추종 직원 위치
- **Topic**: `/pickee/vision/staff_location`  
- **From → To**: Pic Vision → Pic Main  
- **Message Fields**:
```plaintext
int32 robot_id
shopee_interfaces/msg/Point2D relative_position   # 로봇 기준 상대 위치 (m)
float32 distance
bool is_tracking
```

---

### 👷 직원 등록 결과
- **Topic**: `/pickee/vision/register_staff_result`  
- **From → To**: Pic Vision → Pic Main  
- **Message Fields**:
```plaintext
int32 robot_id
bool success
string message
```

---

## 🛠️ 서비스 (Services)

---

### 🧾 매대 상품 인식 요청
- **Service**: `/pickee/vision/detect_products`  
- **From → To**: Pic Main → Pic Vision

#### Request:
```plaintext
int32 robot_id
int32 order_id
int32[] product_ids
```

#### Response:
```plaintext
bool success
string message
```

---

### 🧺 장바구니 내 특정 상품 확인 요청
- **Service**: `/pickee/vision/check_product_in_cart`  
- **From → To**: Pic Main → Pic Vision

#### Request:
```plaintext
int32 robot_id
int32 order_id
int32 product_id
```

#### Response:
```plaintext
bool success
string message
```

---

### 🛒 장바구니 존재 확인 요청
- **Service**: `/pickee/vision/check_cart_presence`  
- **From → To**: Pic Main → Pic Vision

#### Request:
```plaintext
int32 robot_id
int32 order_id
```

#### Response:
```plaintext
bool success
bool cart_present
float32 confidence    # 픽커 비전: 0.0 또는 기본값
string message
```

---

### 🎥 영상 송출 시작 명령
- **Service**: `/pickee/vision/video_stream_start`  
- **From → To**: Pic Main → Pic Vision

#### Request:
```plaintext
string user_type
string user_id
int32 robot_id
```

#### Response:
```plaintext
bool success
string message
```

---

### ⏹️ 영상 송출 중지 명령
- **Service**: `/pickee/vision/video_stream_stop`  
- **From → To**: Pic Main → Pic Vision

#### Request:
```plaintext
string user_type
string user_id
int32 robot_id
```

#### Response:
```plaintext
bool success
string message
```

---

### 🧑 직원 등록 요청
- **Service**: `/pickee/vision/register_staff`  
- **From → To**: Pic Main → Pic Vision

#### Request:
```plaintext
int32 robot_id
```

#### Response:
```plaintext
bool accepted     # 작업 접수 여부
string message
```

---

### 👣 직원 추종 제어
- **Service**: `/pickee/vision/track_staff`  
- **From → To**: Pic Main → Pic Vision

#### Request:
```plaintext
int32 robot_id
bool track   # true: 추종 시작 / false: 추종 중지
```

#### Response:
```plaintext
bool success
string message
```

---

### 🎛️ Vision 모드 설정
- **Service**: `/pickee/vision/set_mode`  
- **From → To**: Pic Main → Pic Vision

#### Request:
```plaintext
int32 robot_id
string mode
```

#### Response:
```plaintext
bool success
string message
```

**mode 종류**:
- `idle`
- `navigation`
- `register_staff`
- `detect_products`
- `track_staff`

---

### 🔈 음성 송출 요청
- **Service**: `/pickee/tts_request`  
- **From → To**: Pic Vision → Pic Main

#### Request:
```plaintext
string text_to_speak
```

#### Response:
```plaintext
bool success
string message
```
