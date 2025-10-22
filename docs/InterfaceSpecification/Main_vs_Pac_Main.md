Main = Shopee Main Service

Pac Main = Packee Main Controller

### `/packee/packing_complete`
> **ROS2 Interface:** `shopee_interfaces/msg/PackeePackingComplete.msg`

### `/packee/robot_status`
> **ROS2 Interface:** `shopee_interfaces/msg/PackeeRobotStatus.msg`

### `/packee/availability_result`
> **ROS2 Interface:** `shopee_interfaces/msg/PackeeAvailability.msg`

### `/packee/packing/check_availability`
> **ROS2 Interface:** `shopee_interfaces/srv/PackeePackingCheckAvailability.srv`

### `/packee/packing/start`
> **ROS2 Interface:** `shopee_interfaces/srv/PackeePackingStart.srv`




## Topic 인터페이스

| 구분 | 메시지명 | 토픽 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **포장 완료 알림** | `/packee/packing_complete` | Topic | Pac Main | Main | `int32 robot_id`<br>`int32 order_id`<br>`bool success`<br>`int32 packed_items`<br>`string message` | **성공**<br>`robot_id: 1`<br>`order_id: 3`<br>`success: true`<br>`packed_items: 5`<br>`message: "Packing completed"`<br><br>**실패**<br>`robot_id: 1`<br>`order_id: 3`<br>`success: false`<br>`packed_items: 3`<br>`message: "Packing failed - gripper error"` |
| **로봇 상태 전송** | `/packee/robot_status` | Topic | Pac Main | Main | `int32 robot_id`<br>`string state`<br>`int32 current_order_id`<br>`int32 items_in_cart` | `robot_id: 1`<br>`state: "packing"`<br>`current_order_id: 3`<br>`items_in_cart: 5` |
| **작업 가능 확인 완료** | `/packee/availability_result` | Topic | Pac Main | Main | `int32 robot_id`<br>`int32 order_id`<br>`bool available`<br>`bool cart_detected`<br>`string message` | **작업 가능**<br>`robot_id: 1`<br>`order_id: 3`<br>`available: true`<br>`cart_detected: true`<br>`message: "Ready for packing"`<br><br>**작업 불가 - 장바구니 없음**<br>`robot_id: 1`<br>`order_id: 3`<br>`available: false`<br>`cart_detected: false`<br>`message: "Cart not detected"`<br><br>**작업 불가 - 로봇 상태**<br>`robot_id: 1`<br>`order_id: 3`<br>`available: false`<br>`cart_detected: true`<br>`message: "Robot busy with another order"` |

## Service 인터페이스

| 구분 | 서비스명 | 서비스 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **작업 가능 확인 요청** | `/packee/packing/check_availability` | Service | Main | Pac Main | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br><br>**Response**<br>`bool success`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br><br>**Response**<br>`success: true`<br>`message: "Availability check initiated"` |
| **포장 시작 명령** | `/packee/packing/start` | Service | Main | Pac Main | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`shopee_interfaces/msg/ProductInfo[] products`<br><br>**Response**<br>`int32 box_id`<br>`bool success`<br>`string message`<br><br>**참고: ProductInfo 구조**<br>`int32 product_id`<br>`int32 quantity`<br>`int32 length`<br>`int32 width`<br>`int32 height`<br>`int32 weight`<br>`bool fragile` | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br>`products: [`<br>  `{ product_id: 101, quantity: 2, ... },`<br>  `{ product_id: 105, quantity: 1, ... }`<br>`]`<br><br>**Response**<br>`box_id: 123`<br>`success: true`<br>`message: "Packing started"` |
