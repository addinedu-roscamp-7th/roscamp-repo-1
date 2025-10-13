Main = Shopee Main Service

Pic Main = Pickee Main Controller

## ROS 인터페이스 매핑

| 분류 | 토픽/서비스 | ROS 타입 | 파일 경로 |
|---|---|---|---|
| Topic | `/pickee/moving_status` | `shopee_interfaces/msg/PickeeMoveStatus` | `ros2_ws/src/shopee_interfaces/msg/PickeeMoveStatus.msg` |
| Topic | `/pickee/arrival_notice` | `shopee_interfaces/msg/PickeeArrival` | `ros2_ws/src/shopee_interfaces/msg/PickeeArrival.msg` |
| Topic | `/pickee/product_detected` | `shopee_interfaces/msg/PickeeProductDetection` | `ros2_ws/src/shopee_interfaces/msg/PickeeProductDetection.msg` |
| Topic | `/pickee/cart_handover_complete` | `shopee_interfaces/msg/PickeeCartHandover` | `ros2_ws/src/shopee_interfaces/msg/PickeeCartHandover.msg` |
| Topic | `/pickee/robot_status` | `shopee_interfaces/msg/PickeeRobotStatus` | `ros2_ws/src/shopee_interfaces/msg/PickeeRobotStatus.msg` |
| Topic | `/pickee/product/selection_result` | `shopee_interfaces/msg/PickeeProductSelection` | `ros2_ws/src/shopee_interfaces/msg/PickeeProductSelection.msg` |
| Topic | `/pickee/product/loaded` | `shopee_interfaces/msg/PickeeProductLoaded` | `ros2_ws/src/shopee_interfaces/msg/PickeeProductLoaded.msg` |
| Service | `/pickee/workflow/start_task` | `shopee_interfaces/srv/PickeeWorkflowStartTask` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowStartTask.srv` |
| Service | `/pickee/workflow/move_to_section` | `shopee_interfaces/srv/PickeeWorkflowMoveToSection` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowMoveToSection.srv` |
| Service | `/pickee/product/detect` | `shopee_interfaces/srv/PickeeProductDetect` | `ros2_ws/src/shopee_interfaces/srv/PickeeProductDetect.srv` |
| Service | `/pickee/product/process_selection` | `shopee_interfaces/srv/PickeeProductProcessSelection` | `ros2_ws/src/shopee_interfaces/srv/PickeeProductProcessSelection.srv` |
| Service | `/pickee/workflow/end_shopping` | `shopee_interfaces/srv/PickeeWorkflowEndShopping` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowEndShopping.srv` |
| Service | `/pickee/workflow/move_to_packaging` | `shopee_interfaces/srv/PickeeWorkflowMoveToPackaging` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowMoveToPackaging.srv` |
| Service | `/pickee/workflow/return_to_base` | `shopee_interfaces/srv/PickeeWorkflowReturnToBase` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowReturnToBase.srv` |
| Service | `/pickee/workflow/return_to_staff` | `shopee_interfaces/srv/PickeeWorkflowReturnToStaff` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowReturnToStaff.srv` |
| Service | `/pickee/video_stream/start` | `shopee_interfaces/srv/PickeeMainVideoStreamStart` | `ros2_ws/src/shopee_interfaces/srv/PickeeMainVideoStreamStart.srv` |
| Service | `/pickee/video_stream/stop` | `shopee_interfaces/srv/PickeeMainVideoStreamStop` | `ros2_ws/src/shopee_interfaces/srv/PickeeMainVideoStreamStop.srv` |
| Service | `/main/get_product_location` | `shopee_interfaces/srv/MainGetProductLocation` | `ros2_ws/src/shopee_interfaces/srv/MainGetProductLocation.srv` |
| Service | `/main/get_location_pose` | `shopee_interfaces/srv/MainGetLocationPose` | `ros2_ws/src/shopee_interfaces/srv/MainGetLocationPose.srv` |
| Service | `/main/get_warehouse_pose` | `shopee_interfaces/srv/MainGetWarehousePose` | `ros2_ws/src/shopee_interfaces/srv/MainGetWarehousePose.srv` |
| Service | `/main/get_section_pose` | `shopee_interfaces/srv/MainGetSectionPose` | `ros2_ws/src/shopee_interfaces/srv/MainGetSectionPose.srv` |

## 인터페이스 상세 정의

| 구분 | 메시지/서비스명 | 토픽/서비스 | From | To | 메시지 구조 |
|---|---|---|---|---|---|
| **Topic** |
| 이동 시작 알림 | `/pickee/moving_status` | Pic Main | Main | `int32 robot_id`<br>`int32 order_id`<br>`int32 location_id` |
| 도착 보고 | `/pickee/arrival_notice` | Pic Main | Main | `int32 robot_id`<br>`int32 order_id`<br>`int32 location_id`<br>`int32 section_id` |
| 상품 위치 인식 완료 | `/pickee/product_detected` | Pic Main | Main | `int32 robot_id`<br>`int32 order_id`<br>`DetectedProduct[] products`<br><br>`# DetectedProduct`<br>`int32 product_id`<br>`int32 bbox_number`<br>`BBox bbox_coords`<br>`float32 confidence`<br><br>`# BBox`<br>`int32 x1`<br>`int32 y1`<br>`int32 x2`<br>`int32 y2` |
| 장바구니 교체 완료 | `/pickee/cart_handover_complete` | Pic Main | Main | `int32 robot_id`<br>`int32 order_id` |
| 로봇 상태 전송 | `/pickee/robot_status` | Pic Main | Main | `int32 robot_id`<br>`string state # Pickee 상태 코드 (예: "PK_S10")`<br>`float32 battery_level`<br>`int32 current_order_id`<br>`float32 position_x`<br>`float32 position_y`<br>`float32 orientation_z` |
| 담기 완료 보고 | `/pickee/product/selection_result` | Pic Main | Main | `int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br>`bool success`<br>`int32 quantity`<br>`string message` |
| 창고 물품 적재 완료 보고 | `/pickee/product/loaded` | Pic Main | Main | `int32 robot_id`<br>`int32 product_id`<br>`int32 quantity`<br>`bool success`<br>`string message` |
| **Service** |
| 작업 시작 명령 | `/pickee/workflow/start_task` | Main | Pic Main | `# Request`<br>`int32 robot_id`<br>`int32 order_id`<br>`string user_id`<br>`ProductLocation[] product_list`<br>`---`<br>`# Response`<br>`bool success`<br>`string message`<br><br>`# ProductLocation`<br>`int32 product_id`<br>`int32 location_id`<br>`int32 section_id`<br>`int32 quantity` |
| 섹션 이동 명령 | `/pickee/workflow/move_to_section` | Main | Pic Main | `# Request`<br>`int32 robot_id`<br>`int32 order_id`<br>`int32 location_id`<br>`int32 section_id`<br>`---`<br>`# Response`<br>`bool success`<br>`string message` |
| 상품 인식 명령 | `/pickee/product/detect` | Main | Pic Main | `# Request`<br>`int32 robot_id`<br>`int32 order_id`<br>`int32[] product_ids`<br>`---`<br>`# Response`<br>`bool success`<br>`string message` |
| 상품 담기 명령 | `/pickee/product/process_selection` | Main | Pic Main | `# Request`<br>`int32 robot_id`<br>`int32 order_id`<br>`int32 product_id`<br>`int32 bbox_number`<br>`---`<br>`# Response`<br>`bool success`<br>`string message` |
| 쇼핑 종료 명령 | `/pickee/workflow/end_shopping` | Main | Pic Main | `# Request`<br>`int32 robot_id`<br>`int32 order_id`<br>`---`<br>`# Response`<br>`bool success`<br>`string message` |
| 포장대 이동 명령 | `/pickee/workflow/move_to_packaging` | Main | Pic Main | `# Request`<br>`int32 robot_id`<br>`int32 order_id`<br>`int32 location_id`<br>`---`<br>`# Response`<br>`bool success`<br>`string message` |
| 복귀 명령 | `/pickee/workflow/return_to_base` | Main | Pic Main | `# Request`<br>`int32 robot_id`<br>`int32 location_id`<br>`---`<br>`# Response`<br>`bool success`<br>`string message` |
| 직원으로 복귀 명령 | `/pickee/workflow/return_to_staff` | Main | Pic Main | `# Request`<br>`int32 robot_id`<br>`---`<br>`# Response`<br>`bool success`<br>`string message`<br><br>*Pic Main이 마지막으로 추종했던 위치를 기억해서 이 service 수신시 이동 시작* |
| 영상 송출 시작 명령 | `/pickee/video_stream/start` | Main | Pic Main | `# Request`<br>`string user_type`<br>`string user_id`<br>`int32 robot_id`<br>`---`<br>`# Response`<br>`bool success`<br>`string message` |
| 영상 송출 중지 명령 | `/pickee/video_stream/stop` | Main | Pic Main | `# Request`<br>`string user_type`<br>`string user_id`<br>`int32 robot_id`<br>`---`<br>`# Response`<br>`bool success`<br>`string message` |
| 상품 위치 조회 | `/main/get_product_location` | Pic Main | Main | `# Request`<br>`int32 product_id`<br>`---`<br>`# Response`<br>`bool success`<br>`int32 warehouse_id`<br>`int32 section_id`<br>`string message` |
| 좌표 정보 조회 | `/main/get_location_pose` | Pic Main | Main | `# Request`<br>`int32 location_id`<br>`---`<br>`# Response`<br>`shopee_interfaces/Pose2D pose`<br>`bool success`<br>`string message` |
| 창고 좌표 정보 조회 | `/main/get_warehouse_pose` | Pic Main | Main | `# Request`<br>`int32 warehouse_id`<br>`---`<br>`# Response`<br>`shopee_interfaces/Pose2D pose`<br>`bool success`<br>`string message` |
| 섹션 좌표 정보 조회 | `/main/get_section_pose` | Pic Main | Main | `# Request`<br>`int32 section_id`<br>`---`<br>`# Response`<br>`shopee_interfaces/Pose2D pose`<br>`bool success`<br>`string message` |