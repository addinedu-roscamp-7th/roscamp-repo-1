Main = Shopee Main Service

Pic Main = Pickee Main Controller

 

 

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
string customer_id
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
int32 quantity
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
string customer_id
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
string customer_id
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