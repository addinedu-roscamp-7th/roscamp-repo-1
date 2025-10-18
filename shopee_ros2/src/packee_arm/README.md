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
- `/packee/arm/pick_status` (`shopee_interfaces/msg/PackeeArmTaskStatus`)  
  픽업 단계별 상태(`planning`, `approaching`, `grasping`, `lifting`, `done`)를 전송합니다.
- `/packee/arm/place_status` (`shopee_interfaces/msg/PackeeArmTaskStatus`)  
  담기 단계별 상태(`planning`, `approaching`, `moving`, `done`)를 전송합니다.
- `/packee/arm/move_to_pose` (`shopee_interfaces/srv/PackeeArmMoveToPose`)  
  `pose_type` 검증 후 자세 전환 명령을 큐에 등록합니다.
- `/packee/arm/pick_product` (`shopee_interfaces/srv/PackeeArmPickProduct`)  
  Bounding Box/타깃 좌표 검증, CNN 신뢰도 확인 후 시각 서보·그리퍼 제어를 수행합니다.
- `/packee/arm/place_product` (`shopee_interfaces/srv/PackeeArmPlaceProduct`)  
  상품을 보유한 팔에 대해 포장 위치로 이동·해제 작업을 진행합니다.

## 빌드 및 실행
```bash
cd <workspace>
colcon build --packages-select packee_arm --base-paths shopee_ros2/src
source install/setup.bash
```

- Arm 컨트롤러 단독 실행
  ```bash
  ros2 launch packee_arm packee_arm.launch.py
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
