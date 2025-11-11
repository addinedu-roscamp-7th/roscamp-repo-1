# Pickee Main - 테스트 가이드

## Service & Topic List

### 서비스 리스트 : ros2 service list -t
```bash
client /main/get_location_pose [shopee_interfaces/srv/MainGetLocationPose]
client /main/get_product_location [shopee_interfaces/srv/MainGetProductLocation]
client /main/get_section_pose [shopee_interfaces/srv/MainGetSectionPose]
client /main/get_warehouse_pose [shopee_interfaces/srv/MainGetWarehousePose]
client /pickee/arm/move_to_pose [shopee_interfaces/srv/PickeeArmMoveToPose]
client /pickee/arm/pick_product [shopee_interfaces/srv/PickeeArmPickProduct]
client /pickee/arm/place_product [shopee_interfaces/srv/PickeeArmPlaceProduct]
client /pickee/mobile/move_to_location [shopee_interfaces/srv/PickeeMobileMoveToLocation]
client /pickee/mobile/update_global_path [shopee_interfaces/srv/PickeeMobileUpdateGlobalPath]
service /pickee/product/detect [shopee_interfaces/srv/PickeeProductDetect]
service /pickee/product/process_selection [shopee_interfaces/srv/PickeeProductProcessSelection]
service /pickee/tts_request [shopee_interfaces/srv/PickeeTtsRequest]
service /pickee/video_stream/start [shopee_interfaces/srv/PickeeMainVideoStreamStart]
service /pickee/video_stream/stop [shopee_interfaces/srv/PickeeMainVideoStreamStop]
client /pickee/vision/check_cart_presence [shopee_interfaces/srv/PickeeVisionCheckCartPresence]
client /pickee/vision/check_product_in_cart [shopee_interfaces/srv/PickeeVisionCheckProductInCart]
client /pickee/vision/detect_products [shopee_interfaces/srv/PickeeVisionDetectProducts]
client /pickee/vision/register_staff [shopee_interfaces/srv/PickeeVisionRegisterStaff]
client /pickee/vision/set_mode [shopee_interfaces/srv/PickeeVisionSetMode]
client /pickee/vision/track_staff [shopee_interfaces/srv/PickeeVisionTrackStaff]
client /pickee/vision/video_stream_start [shopee_interfaces/srv/PickeeVisionVideoStreamStart]
client /pickee/vision/video_stream_stop [shopee_interfaces/srv/PickeeVisionVideoStreamStop]
service /pickee/workflow/end_shopping [shopee_interfaces/srv/PickeeWorkflowEndShopping]
service /pickee/workflow/move_to_packaging [shopee_interfaces/srv/PickeeWorkflowMoveToPackaging]
service /pickee/workflow/move_to_section [shopee_interfaces/srv/PickeeWorkflowMoveToSection]
service /pickee/workflow/return_to_base [shopee_interfaces/srv/PickeeWorkflowReturnToBase]
service /pickee/workflow/return_to_staff [shopee_interfaces/srv/PickeeWorkflowReturnToStaff]
service /pickee/workflow/start_task [shopee_interfaces/srv/PickeeWorkflowStartTask]
```

### 토픽 리스트 : ros2 topic list -t
```bash
sub /pickee/arm/pick_status [shopee_interfaces/msg/PickeeArmTaskStatus]
sub /pickee/arm/place_status [shopee_interfaces/msg/PickeeArmTaskStatus]
sub /pickee/arm/pose_status [shopee_interfaces/msg/ArmPoseStatus]
pub /pickee/arrival_notice [shopee_interfaces/msg/PickeeArrival]
pub /pickee/cart_handover_complete [shopee_interfaces/msg/PickeeCartHandover]
sub /pickee/mobile/arrival [shopee_interfaces/msg/PickeeMobileArrival]
sub /pickee/mobile/pose [shopee_interfaces/msg/PickeeMobilePose]
pub /pickee/mobile/speed_control [shopee_interfaces/msg/PickeeMobileSpeedControl]
pub /pickee/moving_status [shopee_interfaces/msg/PickeeMoveStatus]
pub /pickee/product/loaded [shopee_interfaces/msg/PickeeProductLoaded]
pub /pickee/product/selection_result [shopee_interfaces/msg/PickeeProductSelection]
pub /pickee/product_detected [shopee_interfaces/msg/PickeeProductDetection]
pub /pickee/robot_status [shopee_interfaces/msg/PickeeRobotStatus]
sub /pickee/vision/cart_check_result [shopee_interfaces/msg/PickeeVisionCartCheck]
sub /pickee/vision/detection_result [shopee_interfaces/msg/PickeeVisionDetection]
sub /pickee/vision/obstacle_detected [shopee_interfaces/msg/PickeeVisionObstacles]
sub /pickee/vision/register_staff_result [shopee_interfaces/msg/PickeeVisionStaffRegister]
sub /pickee/vision/staff_location [shopee_interfaces/msg/PickeeVisionStaffLocation]
```

## 터미널 명령어 예시

### 🔧 Service Client 명령어들 (Pic Main이 요청하는 서비스들)

#### **Main Service 조회 서비스들**
```bash
# 제품 위치 조회
ros2 service call /main/get_product_location shopee_interfaces/srv/MainGetProductLocation "{product_id: 123}"

# 위치 좌표 조회
ros2 service call /main/get_location_pose shopee_interfaces/srv/MainGetLocationPose "{location_id: 10}"

# 섹션 좌표 조회
ros2 service call /main/get_section_pose shopee_interfaces/srv/MainGetSectionPose "{section_id: 1}"

# 창고 좌표 조회
ros2 service call /main/get_warehouse_pose shopee_interfaces/srv/MainGetWarehousePose "{warehouse_id: 1}"
```

#### **Mobile 제어 서비스들**
```bash
# Mobile 이동 명령
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "{
  robot_id: 1,
  order_id: 123,
  location_id: 5,
  target_pose: {x: -1.59, y: 1.47, theta: 0.0},
}"
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "{
  robot_id: 1,
  order_id: 123,
  location_id: 1,
  target_pose: {x: 0.71, y: 0.40, theta: 1.57},
}"
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "{
  robot_id: 1,
  order_id: 123,
  location_id: 5,
  target_pose: {x: 3.24, y: 2.1, theta: -1.57},
}"

ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "{
  robot_id: 1,
  order_id: 123,
  location_id: 1,
  target_pose: {x: -2.1, y: 1.5, theta: 0.0},
}"

# Global Path 업데이트
ros2 service call /pickee/mobile/update_global_path shopee_interfaces/srv/PickeeMobileUpdateGlobalPath "{
  robot_id: 1,
  order_id: 123,
  location_id: 3,
  global_path: [
    {x: 5.0, y: 2.5, theta: 0.8},
    {x: 10.5, y: 5.2, theta: 1.57}
  ]
}"
```

#### **Arm 제어 서비스들**
```bash
# Arm 자세 변경
ros2 service call /pickee/arm/move_to_pose shopee_interfaces/srv/PickeeArmMoveToPose "{
  robot_id: 1,
  order_id: 123,
  pose_type: 'shelf_view'
}"

# Arm 제품 픽업
ros2 service call /pickee/arm/pick_product shopee_interfaces/srv/ArmPickProduct "{
  robot_id: 1,
  order_id: 123,
  product_id: 33,
  arm_side: 'left',
  pose: {
    x: 1.0,
    y: 1.0,
    z: 1.0,
    rx: 1.0,
    ry: 1.0,
    rz: 1.0
  }
}"

# Arm 제품 놓기
ros2 service call /pickee/arm/place_product shopee_interfaces/srv/PickeeArmPlaceProduct "{
  robot_id: 1,
  order_id: 123,
  product_id: 456
}"
```

#### **Vision 제어 서비스들**
```bash
# Vision 제품 감지 (서비스 요청 + 응답 토픽 모니터링)
# 먼저 응답 토픽 모니터링 시작:
ros2 topic echo /pickee/vision/detection_result &

# 그 다음 서비스 요청:
ros2 service call /pickee/vision/detect_products shopee_interfaces/srv/PickeeVisionDetectProducts "{
  robot_id: 1,
  order_id: 123,
  product_ids: [1, 2, 3]
}"

# 응답 토픽에서 다음과 같은 형식의 메시지가 나타납니다:
# robot_id: 1
# order_id: 123
# success: true
# products:
# - product_id: 1
#   bbox_number: 1
#   bbox_coords:
#     x1: 100
#     y1: 100
#     x2: 200
#     y2: 200
#   confidence: 0.85
# - product_id: 2
#   bbox_number: 2
#   bbox_coords:
#     x1: 150
#     y1: 130
#     x2: 250
#     y2: 230
#   confidence: 0.9
# message: "Detected 3 products"

# Vision 모드 설정
ros2 service call /pickee/vision/set_mode shopee_interfaces/srv/PickeeVisionSetMode "{
  robot_id: 1,
  mode: 'detect_products'
}"

# Vision 직원 추적
ros2 service call /pickee/vision/track_staff shopee_interfaces/srv/PickeeVisionTrackStaff "{
  robot_id: 1,
  track: true,
}"

# Vision 장바구니 존재 확인
ros2 service call /pickee/vision/check_cart_presence shopee_interfaces/srv/PickeeVisionCheckCartPresence "{
  robot_id: 1,
  order_id: 123
}"

# Vision 장바구니 내 제품 확인
ros2 service call /pickee/vision/check_product_in_cart shopee_interfaces/srv/PickeeVisionCheckProductInCart "{
  robot_id: 1,
  order_id: 123,
  product_id: 456
}"

# Vision 직원 등록
ros2 service call /pickee/vision/register_staff shopee_interfaces/srv/PickeeVisionRegisterStaff "{
  robot_id: 1
}"

# Vision 비디오 스트림 시작
ros2 service call /pickee/vision/video_stream_start shopee_interfaces/srv/PickeeVisionVideoStreamStart "{
  user_type: 'admin',
  user_id: 'admin01',
  robot_id: 1
}"

# Vision 비디오 스트림 중지
ros2 service call /pickee/vision/video_stream_stop shopee_interfaces/srv/PickeeVisionVideoStreamStop "{
  user_type: 'admin',
  user_id: 'admin01',
  robot_id: 1
}"
```

### 🔧 Service Server 테스트 명령어들 (Pic Main이 제공하는 서비스들)

#### **Workflow 서비스들**
```bash
# 작업 시작
ros2 service call /pickee/workflow/start_task shopee_interfaces/srv/PickeeWorkflowStartTask "{
  robot_id: 1,
  order_id: 123,
  user_id: 'USER001',
  product_list: [
    {product_id: 1, location_id: 10, section_id: 1, quantity: 2},
    {product_id: 2, location_id: 15, section_id: 2, quantity: 1}
  ]
}"

# 섹션 이동
ros2 service call /pickee/workflow/move_to_section shopee_interfaces/srv/PickeeWorkflowMoveToSection "{
  robot_id: 1,
  order_id: 123,
  location_id: 1,
  section_id: 1
}"

# 쇼핑 종료
ros2 service call /pickee/workflow/end_shopping shopee_interfaces/srv/PickeeWorkflowEndShopping "{
  robot_id: 1,
  order_id: 123
}"

# 포장대 이동
ros2 service call /pickee/workflow/move_to_packaging shopee_interfaces/srv/PickeeWorkflowMoveToPackaging "{
  robot_id: 1,
  order_id: 123
}"

# 기지로 복귀
ros2 service call /pickee/workflow/return_to_base shopee_interfaces/srv/PickeeWorkflowReturnToBase "{
  robot_id: 1,
  location_id: 123
}"

# 직원에게 복귀
ros2 service call /pickee/workflow/return_to_staff shopee_interfaces/srv/PickeeWorkflowReturnToStaff "{
  robot_id: 1,
}"
```

#### **제품 처리 서비스들**
```bash
# 제품 감지
ros2 service call /pickee/product/detect shopee_interfaces/srv/PickeeProductDetect "{
  robot_id: 1,
  order_id: 123,
  product_ids: [1, 1]
}"

# 제품 선택 처리
ros2 service call /pickee/product/process_selection shopee_interfaces/srv/PickeeProductProcessSelection "{
  robot_id: 1,
  order_id: 123,
  product_id: 456,
  bbox_number: 1
}"
```

#### **영상 및 TTS 서비스들**
```bash
# TTS 요청
ros2 service call /pickee/tts_request shopee_interfaces/srv/PickeeTtsRequest "{
  text_to_speak: '안녕하세요. 테스트 메시지입니다.'
}"

# 비디오 스트림 시작
ros2 service call /pickee/video_stream/start shopee_interfaces/srv/PickeeMainVideoStreamStart "{
  user_type: 'admin',
  user_id: 'admin01',
  robot_id: 1
}"

# 비디오 스트림 중지
ros2 service call /pickee/video_stream/stop shopee_interfaces/srv/PickeeMainVideoStreamStop "{
  user_type: 'admin',
  user_id: 'admin01',
  robot_id: 1
}"
```

### 📡 Publisher 명령어들 (Pic Main이 발행하는 토픽들)

```bash
# Mobile 속도 제어
ros2 topic pub --once /pickee/mobile/speed_control shopee_interfaces/msg/PickeeMobileSpeedControl "{
  robot_id: 1,
  order_id: 44,
  speed_mode: "decelerate",
  target_speed: 0.3,
  obstacles:[
    {
      obstacle_type: "person",
      position: {x: 1.0, y: 1.0},
      distance: 1.5,
      velocity: 0.8,
      bbox: {
        x1: 1,
        y1: 1,
        x2: 2,
        y2: 2,
      },
      confidence: 0.98,
    }
  ],
  reason: "dynamic_obstacle_near"
}"

# 로봇 상태 발행 (자동으로 1Hz로 발행됨, 수동 테스트 불필요)
# ros2 topic echo /pickee/robot_status 로 모니터링

# 도착 알림 발행 (자동 발행, 수동 테스트 불필요)
# ros2 topic echo /pickee/arrival_notice 로 모니터링

# 이동 상태 발행 (자동 발행, 수동 테스트 불필요)
# ros2 topic echo /pickee/moving_status 로 모니터링
```

### 📡 Subscriber 모니터링 명령어들 (Pic Main이 수신하는 토픽들)

```bash
# Arm 상태 모니터링
ros2 topic echo /pickee/arm/pick_status
ros2 topic echo /pickee/arm/place_status  
ros2 topic echo /pickee/arm/pose_status

# Mobile 상태 모니터링
ros2 topic echo /pickee/mobile/arrival
ros2 topic echo /pickee/mobile/pose

# Vision 결과 모니터링
ros2 topic echo /pickee/vision/detection_result         #
ros2 topic echo /pickee/vision/cart_check_result        # 되지만 필요 없음
ros2 topic echo /pickee/vision/obstacle_detected        #
ros2 topic echo /pickee/vision/staff_location           # set_mode 선행
ros2 topic echo /pickee/vision/register_staff_result    # 
```

## 🎯 실용적인 테스트 시나리오

### **시나리오 1: 완전한 쇼핑 워크플로우**
```bash
# 1. 상태 모니터링 시작
ros2 topic echo /pickee/robot_status &

# 2. 작업 시작
ros2 service call /pickee/workflow/start_task shopee_interfaces/srv/PickeeWorkflowStartTask "{robot_id: 1, order_id: 123, user_id: 'TEST_USER', product_list: [{product_id: 1, location_id: 10, section_id: 1, quantity: 2}]}"

# 3. 섹션으로 이동
ros2 service call /pickee/workflow/move_to_section shopee_interfaces/srv/PickeeWorkflowMoveToSection "{robot_id: 1, order_id: 123, section_id: 1}"

# 4. 제품 감지
ros2 service call /pickee/product/detect shopee_interfaces/srv/PickeeProductDetect "{robot_id: 1, order_id: 123, product_ids: [1]}"

# 5. 쇼핑 종료
ros2 service call /pickee/workflow/end_shopping shopee_interfaces/srv/PickeeWorkflowEndShopping "{robot_id: 1, order_id: 123}"
```

### **시나리오 2: Mobile 제어 테스트**
```bash
# 1. Mobile 상태 모니터링
ros2 topic echo /pickee/mobile/pose &
ros2 topic echo /pickee/mobile/arrival &

# 2. 이동 명령
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "{robot_id: 1, order_id: 123, location_id: 3, target_pose: {x: 5.0, y: 3.0, theta: 0.0}, global_path: [], navigation_mode: 'normal'}"

# 3. 속도 제어
ros2 topic pub --once /pickee/mobile/speed_control shopee_interfaces/msg/PickeeMobileSpeedControl "{robot_id: 1, order_id: 123, speed_mode: 'normal', target_speed: 1.0, obstacles: [], reason: 'normal_operation'}"
```

### **시나리오 3: Vision 시스템 테스트**
```bash
# 1. Vision 결과 모니터링 시작
ros2 topic echo /pickee/vision/detection_result &

# 2. Vision 모드 설정
ros2 service call /pickee/vision/set_mode shopee_interfaces/srv/PickeeVisionSetMode "{robot_id: 1, mode: 'track_staff'}"

# 3. 제품 감지 요청 (1초 후 PickeeDetectedProduct 배열이 포함된 응답 토픽 발행됨)
ros2 service call /pickee/vision/detect_products shopee_interfaces/srv/PickeeVisionDetectProducts "{robot_id: 1, order_id: 123, product_ids: [1, 2, 3]}"

# 4. 감지된 제품 정보 확인 (BBox 좌표, confidence 등 포함)
# products 배열에 PickeeDetectedProduct 메시지들이 표시됨
```

---

### 토픽 전체 테스트
``` bash
ros2 topic pub --once /pickee/arm/pick_status shopee_interfaces/msg/PickeeArmTaskStatus "{
  robot_id: 1, 
  order_id: 1,
  product_id: 10,
  status: 'completed',
  current_phase: 'pick_operation',
  progress: 1.0,
  message: 'Test pick completed'
}"

ros2 topic pub --once /pickee/arm/place_status shopee_interfaces/msg/PickeeArmTaskStatus "{
  robot_id: 1, 
  order_id: 1,
  product_id: 10,
  status: 'completed',
  current_phase: 'planning',
  progress: 1.0,
  message: 'Test place completed'
}"

ros2 topic echo /pickee/arm/pose_status
ros2 topic pub --once /pickee/arm/pose_status shopee_interfaces/msg/ArmPoseStatus "{
  robot_id: 1,
  order_id: 1,
  pose_type: 'shelf_view',
  status: 'completed',
  progress: '1.0',
  message: 'copleted'
}"

ros2 topic echo /pickee/arrival_notice
ros2 topic pub --once /pickee/arrival_notice shopee_interfaces/msg/PickeeArrival "{
  robot_id: 1,
  order_id: 1,
  location_id: 1,
  section_id: 1
}"

ros2 topic echo /pickee/cart_handover_complete
ros2 topic pub --once /pickee/cart_handover_complete shopee_interfaces/msg/PickeeCartHandover "{
  robot_id: 1,
  order_id: 1
}"

ros2 topic pub --once /pickee/mobile/arrival shopee_interfaces/msg/PickeeMobileArrival "{
  robot_id: 1,
  order_id: 1,
  location_id: 1,
  final_pose: {
    x: 1.0,
    y: 1.0,
    theta: 1.0
  },
  position_error: 1.0,
  travel_time: 1.0,
  message: 'complete'
}"

# status : 'idle', 'moving', 'stopped', 'charging', 'error'
ros2 topic pub --once /pickee/mobile/pose shopee_interfaces/msg/PickeeMobilePose "{
  robot_id: 1,
  order_id: 1,
  current_pose: {
    x: 1.0,
    y: 1.0,
    theta: 1.0
  },
  linear_velocity: 1.0,
  angular_velocity: 1.0,
  battery_level: 70.0,
  status: 'moving'  
}"

ros2 topic pub --once /pickee/mobile/speed_control shopee_interfaces/msg/PickeeMobileSpeedControl "{
  robot_id: 1
  order_id: 44
  speed_mode: "decelerate"
  target_speed: 0.3
  obstacles:
    - obstacle_type: "person"
      distance: 1.5
      velocity: 0.8
  reason: "dynamic_obstacle_near"
}"

ros2 topic echo /pickee/moving_status
ros2 topic pub --once /pickee/moving_status shopee_interfaces/msg/PickeeMoveStatus "{
  robot_id: 1,
  order_id: 1,
  location_id: 1,
}"

ros2 topic echo /pickee/product/loaded
ros2 topic pub --once /pickee/product/loaded shopee_interfaces/msg/PickeeProductLoaded "{
  robot_id: 1,
  product_id: 1,
  quantity: 5,
  success: true,
  message: 'completed'
}"

ros2 topic echo /pickee/product/selection_result
ros2 topic pub --once /pickee/product/selection_result shopee_interfaces/msg/PickeeProductSelection " {
  robot_id: 1,
  order_id: 1,
  product_id: 10,
  success: true,
  quantity: 5,
  message: 'completed'
}"

ros2 topic echo /pickee/product_detected
ros2 topic pub --once /pickee/product_detected shopee_interfaces/msg/PickeeProductDetection "{
  robot_id: 1,
  order_id: 1,
  products: [{
    product_id: 10,
    bbox_number: 1,
    bbox_coords: {
      x1: 1,
      y1: 1,
      x2: 1,
      y2: 1,
    },
    confidence: 1.0
  }]
}"

ros2 topic echo /pickee/robot_status
ros2 topic pub /pickee/robot_status shopee_interfaces/msg/PickeeRobotStatus "{
  robot_id: 1,
  state: 'PK_S10',
  battery_level: 70.0,
  current_order_id: 1,
  position_x: 0.0,
  position_y: 0.0,
  orientation_z: 0.0
}"

ros2 topic echo /pickee/vision/cart_check_result
ros2 topic pub --once /pickee/vision/cart_check_result shopee_interfaces/msg/PickeeVisionCartCheck "{
  robot_id: 1,
  order_id: 4,
  success: true,
  product_id: 5,
  found: true,
  quantity: 2,
  message: 'Product found in cart'
}"

ros2 topic echo /pickee/vision/detection_result
ros2 topic pub --once /pickee/vision/detection_result shopee_interfaces/msg/PickeeVisionDetection "{
  robot_id: 1,
  order_id: 4,
  success: true,
  products: [
    {
      product_id: 4,
      bbox_number: 1,
      bbox_coords: {x1: 100, y1: 150, x2: 200, y2: 250},
      confidence: 0.95
    },
    {
      product_id: 5,
      bbox_number: 2,
      bbox_coords: {x1: 250, y1: 150, x2: 350, y2: 250},
      confidence: 0.92
    }
  ],
  message: '2 products detected'
}"

ros2 topic echo /pickee/vision/obstacle_detected
ros2 topic pub --once /pickee/vision/obstacle_detected shopee_interfaces/msg/PickeeVisionObstacles "{
  robot_id: 1,
  order_id: 4,
  obstacles: [
    {
      obstacle_type: 'cart',
      position: {x: 5.2, y: 3.1},
      distance: 2.5,
      velocity: 0.0,
      direction: {vx: 0.0, vy: 0.0},
      bbox: {x1: 200, y1: 150, x2: 350, y2: 400},
      confidence: 0.92,
    }
  ],
  message: '1 static obstacle detected'
}"

ros2 topic echo /pickee/vision/register_staff_result
ros2 topic pub --once /pickee/vision/register_staff_result shopee_interfaces/msg/PickeeVisionStaffRegister "{
  robot_id: 1,
  success: true,
  message: 'Staff registration successful.'
}"

ros2 topic echo /pickee/vision/staff_location
ros2 topic pub --once /pickee/vision/staff_location shopee_interfaces/msg/PickeeVisionStaffLocation "{
  robot_id: 1,
  relative_position: {x: 2.5, y: 0.3},
  distance: 2.52,
  is_tracking: true
}"
```



### 전체 연동 테스트
``` bash
ros2 run main_service mock_llm_server
ros2 run main_service mock_packee_node 
ros2 run main_service main_service_node 
python3 src/main_service/scripts/test_client.py 
export PYTHONPATH=$PYTHONPATH:/home/wonho/venv/ros_venv/lib/python3.12/site-packages
# db 세팅
shopee_ros2/src/main_service/scripts/setup_database.sh
shopee_ros2/src/main_service/scripts/reset_database.sh
```