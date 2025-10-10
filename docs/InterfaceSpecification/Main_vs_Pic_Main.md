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
| Service | `/pickee/workflow/start_task` | `shopee_interfaces/srv/PickeeWorkflowStartTask` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowStartTask.srv` |
| Service | `/pickee/workflow/move_to_section` | `shopee_interfaces/srv/PickeeWorkflowMoveToSection` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowMoveToSection.srv` |
| Service | `/pickee/product/detect` | `shopee_interfaces/srv/PickeeProductDetect` | `ros2_ws/src/shopee_interfaces/srv/PickeeProductDetect.srv` |
| Service | `/pickee/product/process_selection` | `shopee_interfaces/srv/PickeeProductProcessSelection` | `ros2_ws/src/shopee_interfaces/srv/PickeeProductProcessSelection.srv` |
| Service | `/pickee/workflow/end_shopping` | `shopee_interfaces/srv/PickeeWorkflowEndShopping` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowEndShopping.srv` |
| Service | `/pickee/workflow/move_to_packaging` | `shopee_interfaces/srv/PickeeWorkflowMoveToPackaging` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowMoveToPackaging.srv` |
| Service | `/pickee/workflow/return_to_base` | `shopee_interfaces/srv/PickeeWorkflowReturnToBase` | `ros2_ws/src/shopee_interfaces/srv/PickeeWorkflowReturnToBase.srv` |
| Service | `/pickee/video_stream/start` | `shopee_interfaces/srv/PickeeMainVideoStreamStart` | `ros2_ws/src/shopee_interfaces/srv/PickeeMainVideoStreamStart.srv` |
| Service | `/pickee/video_stream/stop` | `shopee_interfaces/srv/PickeeMainVideoStreamStop` | `ros2_ws/src/shopee_interfaces/srv/PickeeMainVideoStreamStop.srv` |
| Service | `/main/get_product_location` | `shopee_interfaces/srv/MainGetProductLocation` | `ros2_ws/src/shopee_interfaces/srv/MainGetProductLocation.srv` |

 

From

To

Message

 

 

From

To

Message

Topic

이동 시작 알림

 

/pickee/moving_status

Pic Main

Main



int32 robot_id
int order_id
int location_id
도착 보고

 

/pickee/arrival_notice

Pic Main

Main



int32 robot_id
int order_id
int location_id
int section_id
상품 위치 인식 완료

 

/pickee/product_detected

Pic Main

Main



int32 robot_id
int order_id
DetectedProduct[] products


# DetectedProduct
int product_id
int32 bbox_number
BBox bbox_coords
float32 confidence


# BBox
int32 x1
int32 y1
int32 x2
int32 y2
장바구니 교체 완료

/pickee/cart_handover_complete

Pic Main

Main



int32 robot_id
int order_id
로봇 상태 전송

/pickee/robot_status

Pic Main

Main



int32 robot_id
string state # Pickee 상태 코드 (예: "PK_S10")
float32 battery_level
int current_order_id
float32 position_x
float32 position_y
float32 orientation_z
담기 완료 보고

/pickee/product/selection_result

Pic Main

Main



int32 robot_id
int order_id
int product_id
bool success
int32 quantity
string message
Service

작업 시작 명령

 

/pickee/workflow/start_task

Main

Pic Main



# Request
int32 robot_id
int order_id
string user_id
ProductLocation[] product_list
---
# Response
bool success
string message


# ProductLocation
int product_id
int location_id
int section_id
int32 quantity
섹션 이동 명령

 

/pickee/workflow/move_to_section

Main

Pic Main



# Request
int32 robot_id
int order_id
int location_id
int section_id
---
# Response
bool success
string message
상품 인식 명령

/pickee/product/detect

Main

Pic Main



# Request
int32 robot_id
int order_id
int[] product_ids
---
# Response
bool success
string message
상품 담기 명령

/pickee/product/process_selection

Main

Pic Main



# Request
int32 robot_id
int order_id
int product_id
int32 bbox_number
---
# Response
bool success
string message
쇼핑 종료 명령

/pickee/workflow/end_shopping

Main

Pic Main



# Request
int32 robot_id
int order_id
---
# Response
bool success
string message
포장대 이동 명령

/pickee/workflow/move_to_packaging

Main

Pic Main



# Request
int32 robot_id
int order_id
int location_id
---
# Response
bool success
string message
복귀 명령

/pickee/workflow/return_to_base

Main

Pic Main



# Request
int32 robot_id
int location_id
---
# Response
bool success
string message
영상 송출 시작 명령

/pickee/video_stream/start

Main

Pic Main



# Request
string user_type
string user_id
int32 robot_id
---
# Response
bool success
string message
영상 송출 중지 명령

/pickee/video_stream/stop

Main

Pic Main



# Request
string user_type
string user_id
int32 robot_id
---
# Response
bool success
string message
​상품 위치 조회

/main/get_product_location

Pic Main

Main



# Request
int product_id
---
# Response
bool success
int warehouse_id
int section_id
string message
