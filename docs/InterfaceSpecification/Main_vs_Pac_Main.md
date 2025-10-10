Main = Shopee Main Service

Pac Main = Packee Main Controller

## ROS 인터페이스 매핑

| 분류 | 토픽/서비스 | ROS 타입 |
|---|---|---|
| Topic | `/packee/packing_complete` | `shopee_interfaces/msg/PackeePackingComplete` |
| Topic | `/packee/robot_status` | `shopee_interfaces/msg/PackeeRobotStatus` |
| Topic | `/packee/availability_result` | `shopee_interfaces/msg/PackeeAvailability` |
| Service | `/packee/packing/check_availability` | `shopee_interfaces/srv/PackeePackingCheckAvailability` |
| Service | `/packee/packing/start` | `shopee_interfaces/srv/PackeePackingStart` |




From

To

Message

예시

Topic











포장 완료 알림

/packee/packing_complete

Pac Main

Main

int32 robot_id
int32 order_id
bool success
int32 packed_items
string message

# 성공
robot_id: 1
order_id: 3
success: true
packed_items: 5
message: "Packing completed"

# 실패
robot_id: 1
order_id: 3
success: false
packed_items: 3
message: "Packing failed - gripper error"

로봇 상태 전송

/packee/robot_status

Pac Main

Main

int32 robot_id
string state
int32 current_order_id
int32 items_in_cart

robot_id: 1
state: "packing"
current_order_id: 3
items_in_cart: 5

작업 가능 확인 완료

/packee/availability_result

Pac Main

Main

int32 robot_id
int32 order_id
bool available
bool cart_detected
string message

# 작업 가능
robot_id: 1
order_id: 3
available: true
cart_detected: true
message: "Ready for packing"

# 작업 불가 - 장바구니 없음
robot_id: 1
order_id: 3
available: false
cart_detected: false
message: "Cart not detected"

# 작업 불가 - 로봇 상태
robot_id: 1
order_id: 3
available: false
cart_detected: true
message: "Robot busy with another order"

Service











작업 가능 확인 요청

/packee/packing/check_availability

Main

Pac Main

# Request
int32 robot_id
int32 order_id

---
# Response
bool success
string message

# Request
robot_id: 1
order_id: 3

# Response
success: true
message: "Availability check initiated"

포장 시작 명령

/packee/packing/start

Main 

Pac Main

# Request
int32 robot_id
int32 order_id

---
# Response
bool success
string message

# Request
robot_id: 1
order_id: 3

# Response
success: true
message: "Packing started"
