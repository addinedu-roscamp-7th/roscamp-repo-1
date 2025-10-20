Main = Shopee Main Service

Pic Main = Pickee Main Controller

### `/pickee/moving_status`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeMoveStatus.msg`

### `/pickee/arrival_notice`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeArrival.msg`

> **참고**: 섹션이 아닌 위치(포장대, 대기 영역 등)에 도착한 경우 `section_id`는 `-1`로 전달됩니다.

### `/pickee/product_detected`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeProductDetection.msg`

### `/pickee/cart_handover_complete`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeCartHandover.msg`

### `/pickee/robot_status`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeRobotStatus.msg`

### `/pickee/product/selection_result`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeProductSelection.msg`

### `/pickee/product/loaded`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeProductLoaded.msg`

### `/pickee/workflow/start_task`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeWorkflowStartTask.srv`

### `/pickee/workflow/move_to_section`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeWorkflowMoveToSection.srv`

### `/pickee/product/detect`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeProductDetect.srv`

### `/pickee/product/process_selection`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeProductProcessSelection.srv`

### `/pickee/workflow/end_shopping`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeWorkflowEndShopping.srv`

### `/pickee/workflow/move_to_packaging`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeWorkflowMoveToPackaging.srv`

### `/pickee/workflow/return_to_base`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeWorkflowReturnToBase.srv`

### `/pickee/workflow/return_to_staff`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeWorkflowReturnToStaff.srv`

### `/pickee/video_stream/start`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeMainVideoStreamStart.srv`

### `/pickee/video_stream/stop`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeMainVideoStreamStop.srv`

### `/main/get_product_location`
> **ROS2 Interface:** `shopee_interfaces/srv/MainGetProductLocation.srv`

### `/main/get_location_pose`
> **ROS2 Interface:** `shopee_interfaces/srv/MainGetLocationPose.srv`

### `/main/get_warehouse_pose`
> **ROS2 Interface:** `shopee_interfaces/srv/MainGetWarehousePose.srv`

### `/main/get_section_pose`
> **ROS2 Interface:** `shopee_interfaces/srv/MainGetSectionPose.srv`

## 인터페이스 상세 정의

## 📦 메시지 (Messages)

---

### 🚚 이동 시작 알림
- **Topic**: `/pickee/moving_status`  
- **Message Type**: `shopee_interfaces/msg/PickeeMoveStatus.msg`  
- **From → To**: Pic Main → Main  
- **Fields**:
  ```plaintext
  int32 robot_id
  int32 order_id
  int32 location_id
  ```

---

### 📍 도착 보고
- **Topic**: `/pickee/arrival_notice`  
- **Message Type**: `shopee_interfaces/msg/PickeeArrival.msg`  
- **From → To**: Pic Main → Main  
- **Fields**:
  ```plaintext
  int32 robot_id
  int32 order_id
  int32 location_id
  int32 section_id  # section이 아닌 경우 section_id = -1
  ```

---

### 🔍 상품 위치 인식 완료
- **Topic**: `/pickee/product_detected`  
- **Message Type**: `shopee_interfaces/msg/PickeeProductDetection.msg`  
- **From → To**: Pic Main → Main  
- **Fields**:
  ```plaintext
  int32 robot_id
  int32 order_id
  DetectedProduct[] products
  ```

- **DetectedProduct** (Pickee 사용 필드)
  ```plaintext
  int32 product_id
  float32 confidence
  BBox bbox
  int32 bbox_number       # 앱 UI 선택용
  DetectionInfo detection_info
  Point3D position        # (0, 0, 0) 미사용
  ```

- **DetectionInfo**
  ```plaintext
  Point2D[] polygon   # polygon: 다각형 꼭짓점 좌표 리스트
  BBox bbox_coords
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

📝 *2025.10.20 - DetectionInfo로 통합. polygon 좌표 사용.*

---

### 🔄 장바구니 교체 완료
- **Topic**: `/pickee/cart_handover_complete`  
- **Message Type**: `shopee_interfaces/msg/PickeeCartHandover.msg`  
- **From → To**: Pic Main → Main  
- **Fields**:
  ```plaintext
  int32 robot_id
  int32 order_id
  ```

---

### 📡 로봇 상태 전송
- **Topic**: `/pickee/robot_status`  
- **Message Type**: `shopee_interfaces/msg/PickeeRobotStatus.msg`  
- **From → To**: Pic Main → Main  
- **Fields**:
  ```plaintext
  int32 robot_id
  string state             # 예: "PK_S10"
  float32 battery_level
  int32 current_order_id
  float32 position_x
  float32 position_y
  float32 orientation_z
  ```

---

### 🛒 담기 완료 보고
- **Topic**: `/pickee/product/selection_result`  
- **Message Type**: `shopee_interfaces/msg/PickeeProductSelection.msg`  
- **From → To**: Pic Main → Main  
- **Fields**:
  ```plaintext
  int32 robot_id
  int32 order_id
  int32 product_id
  bool success
  int32 quantity
  string message
  ```

---

### 📦 창고 물품 적재 완료
- **Topic**: `/pickee/product/loaded`  
- **Message Type**: `shopee_interfaces/msg/PickeeProductLoaded.msg`  
- **From → To**: Pic Main → Main  
- **Fields**:
  ```plaintext
  int32 robot_id
  int32 product_id
  int32 quantity
  bool success
  string message
  ```

---

## 🔧 서비스 (Services)

---

### ▶️ 작업 시작 명령
- **Service**: `/pickee/workflow/start_task`  
- **Type**: `shopee_interfaces/srv/PickeeWorkflowStartTask.srv`  
- **From → To**: Main → Pic Main

#### Request:
```plaintext
int32 robot_id
int32 order_id
string user_id
ProductLocation[] product_list
```

#### Response:
```plaintext
bool success
string message
```

**ProductLocation:**
```plaintext
int32 product_id
int32 location_id
int32 section_id
int32 quantity
```

---

### 🚶 섹션 이동 명령
- **Service**: `/pickee/workflow/move_to_section`  
- **Type**: `shopee_interfaces/srv/PickeeWorkflowMoveToSection.srv`  
- **From → To**: Main → Pic Main  

#### Request:
```plaintext
int32 robot_id
int32 order_id
int32 location_id
int32 section_id
```

#### Response:
```plaintext
bool success
string message
```

---

### 🔍 상품 인식 명령
- **Service**: `/pickee/product/detect`  
- **Type**: `shopee_interfaces/srv/PickeeProductDetect.srv`  
- **From → To**: Main → Pic Main  

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

### 🛍️ 상품 담기 명령
- **Service**: `/pickee/product/process_selection`  
- **Type**: `shopee_interfaces/srv/PickeeProductProcessSelection.srv`  
- **From → To**: Main → Pic Main  

#### Request:
```plaintext
int32 robot_id
int32 order_id
int32 product_id
int32 bbox_number
```

#### Response:
```plaintext
bool success
string message
```

---

### 🛑 쇼핑 종료 명령
- **Service**: `/pickee/workflow/end_shopping`  
- **Type**: `shopee_interfaces/srv/PickeeWorkflowEndShopping.srv`  
- **From → To**: Main → Pic Main  

#### Request:
```plaintext
int32 robot_id
int32 order_id
```

#### Response:
```plaintext
bool success
string message
```

---

### 📦 포장대 이동 명령
- **Service**: `/pickee/workflow/move_to_packaging`  
- **Type**: `shopee_interfaces/srv/PickeeWorkflowMoveToPackaging.srv`  
- **From → To**: Main → Pic Main  

#### Request:
```plaintext
int32 robot_id
int32 order_id
int32 location_id
```

#### Response:
```plaintext
bool success
string message
```

---

### 🔁 복귀 명령
- **Service**: `/pickee/workflow/return_to_base`  
- **Type**: `shopee_interfaces/srv/PickeeWorkflowReturnToBase.srv`  
- **From → To**: Main → Pic Main  

#### Request:
```plaintext
int32 robot_id
int32 location_id
```

#### Response:
```plaintext
bool success
string message
```

---

### 👤 직원으로 복귀 명령
- **Service**: `/pickee/workflow/return_to_staff`  
- **Type**: `shopee_interfaces/srv/PickeeWorkflowReturnToStaff.srv`  
- **From → To**: Main → Pic Main  

#### Request:
```plaintext
int32 robot_id
```

#### Response:
```plaintext
bool success
string message
```

> 📍 Pic Main이 마지막 추종 위치 기억 → 이 서비스 수신 시 이동 시작

---

## 📹 영상 스트리밍 명령

### 🎥 영상 송출 시작
- **Service**: `/pickee/video_stream/start`  
- **Type**: `shopee_interfaces/srv/PickeeMainVideoStreamStart.srv`  
- **From → To**: Main → Pic Main  

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

### ⏹️ 영상 송출 중지
- **Service**: `/pickee/video_stream/stop`  
- **Type**: `shopee_interfaces/srv/PickeeMainVideoStreamStop.srv`  
- **From → To**: Main → Pic Main  

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

## 🗺️ 위치 조회 서비스

### 상품 위치 조회
- **Service**: `/main/get_product_location`  
- **Type**: `shopee_interfaces/srv/MainGetProductLocation.srv`  
- **From → To**: Pic Main → Main  

#### Request:
```plaintext
int32 product_id
```

#### Response:
```plaintext
bool success
int32 warehouse_id
int32 section_id
string message
```

---

### 📍 좌표 정보 조회 (Location 기준)  
- **Service**: `/main/get_location_pose`  
- **Type**: `shopee_interfaces/srv/MainGetLocationPose.srv`  
- **From → To**: Pic Main → Main

#### Request:
```plaintext
int32 location_id
```

#### Response:
```plaintext
shopee_interfaces/Pose2D pose
bool success
string message
```

---

### 🏢 창고 좌표 정보 조회  
- **Service**: `/main/get_warehouse_pose`  
- **Type**: `shopee_interfaces/srv/MainGetWarehousePose.srv`  
- **From → To**: Pic Main → Main

#### Request:
```plaintext
int32 warehouse_id
```

#### Response:
```plaintext
shopee_interfaces/Pose2D pose
bool success
string message
```

---

### 📦 섹션 좌표 정보 조회  
- **Service**: `/main/get_section_pose`  
- **Type**: `shopee_interfaces/srv/MainGetSectionPose.srv`  
- **From → To**: Pic Main → Main

#### Request:
```plaintext
int32 section_id
```

#### Response:
```plaintext
shopee_interfaces/Pose2D pose
bool success
string message
```

---

## 📐 Pose2D 구조
```plaintext
float32 x
float32 y
float32 theta
```