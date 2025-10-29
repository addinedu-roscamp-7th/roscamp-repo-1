# Shopee Packee Arm Controller

Packee 로봇의 양팔(Dual-Arm) 작업을 담당하는 ROS 2 패키지입니다.  
Packee Main Controller와 `docs/InterfaceSpecification/Pac_Main_vs_Pac_Arm.md`에서 정의된
서비스/토픽 인터페이스를 통해 연동되며, 포장 워크플로우에 필요한 자세 전환·픽업·담기
명령을 실행하고 진행 상태를 상위 노드에 제공합니다.

## 디렉터리 및 파일 역할
- `include/packee_arm/constants.hpp`  
  시각 서보 허용 오차 등 공통 상수를 정의합니다.
- `include/packee_arm/types.hpp`  
  자세 추정, 명령 구조체 등 하위 모듈 간 공유 타입을 선언합니다.
- `include/packee_arm/visual_servo_module.hpp` / `src/visual_servo_module.cpp`  
  두-stream CNN 기반 P 제어를 담당하는 시각 서보 모듈입니다.
- `include/packee_arm/arm_driver_proxy.hpp` / `src/arm_driver_proxy.cpp`  
  하드웨어 속도 명령을 추상화하고 속도 제한·타임아웃을 검사합니다.
- `include/packee_arm/gripper_controller.hpp` / `src/gripper_controller.cpp`  
  그리퍼 힘 제한과 개폐를 관리합니다.
- `include/packee_arm/execution_manager.hpp` / `src/execution_manager.cpp`  
  좌/우 팔 큐를 비동기 처리하며 시각 서보, 그리퍼 제어, 상태 발행을 조정합니다.
- `src/packee_arm_controller.cpp`  
  ROS 노드 진입점으로 서비스 요청 검증, 파라미터 관리, 하위 모듈 초기화를 수행합니다.
- `src/mock_packee_main.cpp`  
  상위 노드 없이 Arm 컨트롤러를 테스트하기 위한 모의 클라이언트입니다.
- `launch/packee_arm.launch.py`  
  Arm 컨트롤러와 필요 시 모의 메인 노드를 한 번에 실행하는 런치 파일입니다.
- `resource/`  
  `ros2 run` 실행 파일 등록용 리소스입니다.

## ROS 인터페이스 요약
- `/packee/arm/pose_status` (`shopee_interfaces/msg/ArmPoseStatus`)  
  자세 이동 진행률과 메시지를 발행합니다.
- `/packee/arm/pick_status` (`shopee_interfaces/msg/ArmTaskStatus`)  
  픽업 단계별 상태(`planning`, `approaching`, `grasping`, `lifting`, `done`)를 전송합니다.
- `/packee/arm/place_status` (`shopee_interfaces/msg/ArmTaskStatus`)  
  담기 단계별 상태(`planning`, `approaching`, `moving`, `done`)를 전송합니다.
- `/packee/arm/move_to_pose` (`shopee_interfaces/srv/ArmMoveToPose`)  
  `pose_type` 검증 후 자세 전환 명령을 큐에 등록합니다.
- `/packee/arm/pick_product` (`shopee_interfaces/srv/ArmPickProduct`)  
  Bounding Box/타깃 좌표 검증, CNN 신뢰도 확인 후 시각 서보·그리퍼 제어를 수행합니다.
- `/packee/arm/place_product` (`shopee_interfaces/srv/ArmPlaceProduct`)  
  상품을 보유한 팔에 대해 포장 위치로 이동·해제 작업을 진행합니다.

## myCobot 280 연동 가이드
- Packee Arm Controller는 Elephant Robotics **myCobot 280** 듀얼 암을 기준으로 기본 파라미터를 설정합니다.  
  - `servo_gain_xy=0.02`, `servo_gain_z=0.018`, `servo_gain_yaw=0.04`  
  - `max_translation_speed=0.05`, `max_yaw_speed_deg=40.0`, `gripper_force_limit=12.0`  
  - `progress_publish_interval=0.15`, `command_timeout_sec=4.0`
- 프리셋 자세는 베이스 원점(미터) 기준으로 설정합니다. 필요 시 Launch 파일에서 파라미터를 오버라이드하세요.  
  - `preset_pose_cart_view=[0.16, 0.0, 0.18, 0.0]`  
  - `preset_pose_standby=[0.10, 0.0, 0.14, 0.0]`
- 안전 작업 공간은 수평 반경 0.28 m, Z 범위 0.05~0.30 m로 제한되며 서비스를 통해 전달되는 `target_product.pose`, `pose`도 동일하게 검증됩니다.
- 실제 하드웨어 제어 전에는 `jetcobot_bridge` 노드가 사용할 시리얼 장치를 확인하고, Jetson 측에서 `pymycobot` 패키지를 설치해야 합니다.  
  ```bash
  sudo apt-get install python3-pip
  pip3 install pymycobot
  ```

### JetCobot Bridge 노드
- `ArmDriverProxy`와 `GripperController`는 각각 `TwistStamped`, `Float32` 메시지를 `/packee/jetcobot/<arm>/cmd_vel`, `/packee/jetcobot/<arm>/gripper_cmd` 토픽으로 발행합니다.
- `scripts/jetcobot_bridge.py`는 위 토픽을 구독해 JetCobot 시리얼 포트로 명령을 전달하며, `ros2 launch packee_arm packee_arm.launch.py run_jetcobot_bridge:=true` 로 함께 기동할 수 있습니다.
- 주요 런치 인자
  - `left_serial_port`, `right_serial_port`: JetCobot USB 포트 경로 (단일 팔인 경우 오른쪽을 비웁니다)
  - `left_arm_velocity_topic`, `right_arm_velocity_topic`: Velocity 명령 토픽 (필요 시 네임스페이스 조정)
  - `left_gripper_topic`, `right_gripper_topic`: 그리퍼 명령 토픽
  - `jetcobot_move_speed`: `sync_send_coords` 속도(0~100), 기본 40
  - `jetcobot_command_period`: 속도 명령 적분 간격, 기본 0.15 초

## 빌드 및 실행
```bash
cd <workspace>
colcon build --packages-select packee_arm --base-paths shopee_ros2/src
source install/setup.bash
```

- Arm 컨트롤러 단독 실행
  ```bash
  ros2 launch packeeros2 run packee_arm jetcobot_subscriber

# 터미널 2
ros2 run packee_arm mock_packee_main_arm packee_arm.launch.py
  ```

- 모의 메인 노드를 함께 실행하려면 `run_mock_main:=true` 인자를 지정합니다.
  ```bash
  ros2 launch packee_arm packee_arm.launch.py run_mock_main:=true
  ```

- 개별 노드를 직접 실행할 수도 있습니다.
  ```bash
  ros2 run packee_arm packee_arm_controller
  ros2 run packee_arm mock_packee_main
  ```

## 개발 메모
- 새로운 포즈·단계가 추가될 경우 `valid_pose_types_` 또는 `ExecutionManager` 상태 매핑을 함께
  갱신해야 합니다.
- 파라미터는 런타임에 갱신 가능하며, Validation 실패 시 원자적으로 거부됩니다.
- Visual Servo 모듈은 CNN 추론부가 교체되더라도 동일한 인터페이스를 유지하도록 설계되었습니다.
