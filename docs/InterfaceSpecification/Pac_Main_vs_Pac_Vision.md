Pac Main = Packee Main Controller

Pac Vision = Packee Vision AI Service

### `/packee/vision/check_cart_presence`
> **ROS2 Interface:** `shopee_interfaces/srv/VisionCheckCartPresence.srv`

### `/packee/vision/detect_products_in_cart`
> **ROS2 Interface:** `shopee_interfaces/srv/PackeeVisionDetectProductsInCart.srv`

### `/packee/vision/verify_packing_complete`
> **ROS2 Interface:** `shopee_interfaces/srv/PackeeVisionVerifyPackingComplete.srv`

**구조체 매핑**
- `DetectedProduct` → `shopee_interfaces/msg/DetectedProduct` (Pickee/Packee 공통)
- `BBox` → `shopee_interfaces/msg/BBox`
- `Point3D` → `shopee_interfaces/msg/Point3D`
- `DetectionInfo` → `shopee_interfaces/msg/DetectionInfo`

**DetectedProduct 필드 사용 규칙 (Packee)**
- 사용 필드: `product_id`, `confidence`, `bbox`, `position`
- 미사용 필드: `bbox_number` (0), `detection_info` (빈 배열)




## Service 인터페이스

| 구분 | 서비스명 | 서비스 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **장바구니 유무 확인** | `/packee/vision/check_cart_presence` | Service | Pac Main | Pac Vision | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br><br>**Response**<br>`bool success`<br>`bool cart_present`<br>`float32 confidence`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br><br>**Response - 장바구니 있음**<br>`success: true`<br>`cart_present: true`<br>`confidence: 0.98`<br>`message: "Cart detected"`<br><br>**Response - 장바구니 없음**<br>`success: true`<br>`cart_present: false`<br>`confidence: 0.95`<br>`message: "No cart detected"` |
| **장바구니 내 상품 위치 확인** | `/packee/vision/detect_products_in_cart` | Service | Pac Main | Pac Vision | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`int32[] expected_product_ids`<br><br>**Response**<br>`bool success`<br>`shopee_interfaces/msg/DetectedProduct[] products`<br>`int32 total_detected`<br>`string message`<br><br>**DetectedProduct** (Packee 사용 필드)<br>`int32 product_id`<br>`float32 confidence`<br>`shopee_interfaces/msg/BBox bbox`<br>`shopee_interfaces/msg/Point3D position` (그리핑 위치)<br>`int32 bbox_number` (0, 미사용)<br>`shopee_interfaces/msg/DetectionInfo detection_info` (빈 배열, 미사용)<br><br>**BBox**<br>`int32 x1, y1, x2, y2`<br><br>**Pose6D**<br>`float32 joint_1, joint_2, joint_3, joint_4, joint_5, joint_6` | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br>`expected_product_ids: [1, 2, 3]`<br><br>**Response - 성공**<br>`success: true`<br>`products:`<br>`- product_id: 3`<br>`  bbox: {x1: 120, y1: 180, x2: 250, y2: 320}`<br>`  confidence: 0.94`<br>`  pose: {joint_1: 0.3, joint_2: 0.15, joint_3: 0.8, joint_4: 0.4, joint_5: 0.3, joint_6: 0.9}`<br>`total_detected: 3`<br>`message: "All products detected"`<br><br>**Response - 일부만 감지**<br>`total_detected: 2`<br>`message: "Detected 2 out of 3 products"`<br><br>**Response - 실패**<br>`success: false`<br>`products: []`<br>`total_detected: 0`<br>`message: "No products detected in cart"` |
| **포장 완료 확인** | `/packee/vision/verify_packing_complete` | Service | Pac Main | Pac Vision | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br><br>**Response**<br>`bool cart_empty`<br>`int32 remaining_items`<br>`int32[] remaining_product_ids`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br><br>**Response - 포장 완료**<br>`cart_empty: true`<br>`remaining_items: 0`<br>`remaining_product_ids: []`<br>`message: "Cart is empty, packing complete"`<br><br>**Response - 일부 남음**<br>`cart_empty: false`<br>`remaining_items: 1`<br>`remaining_product_ids: [3]`<br>`message: "1 item(s) remaining in cart"` |
