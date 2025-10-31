# Shopee Packee Arm Controller

Packee 로봇의 듀얼 myCobot 280 팔을 Python 기반으로 제어하는 ROS 2 패키지입니다.  
`docs/InterfaceSpecification/Pac_Main_vs_Pac_Arm.md`에 정의된 서비스/토픽 규격을 따르며,
좌·우 팔에 대한 자세 전환, 픽업, 담기 명령을 `pymycobot` 라이브러리로 직접 실행합니다.

## 구성 요소
- `scripts/jetcobot_bridge.py`  
  Packee Arm 상위 모듈이 발행하는 속도·그리퍼 명령을 통합해 myCobot 시리얼 명령으로 변환합니다.
- `src/pymycobot_dual`  
  좌/우 팔을 동시에 제어하며, 인터페이스 명세에 정의된 서비스와 토픽을 단일 노드에서 처리합니다.
- `src/pymycobot_left`  
  필요 시 좌측 팔만 독립적으로 제어할 수 있는 경량 노드입니다.
- `src/pymycobot_right`  
  필요 시 우측 팔만 독립적으로 제어할 수 있는 경량 노드입니다.
- `launch/packee_arm.launch.py`  
  듀얼 팔 노드를 기본으로 기동하며, 옵션에 따라 단일 팔 노드와 JetCobot 브릿지를 실행할 수 있습니다.

## 주요 서비스/토픽
- `/packee/arm/move_to_pose` (`shopee_interfaces/srv/ArmMoveToPose`)  
  `pose_type`을 `cart_view` 또는 `standby`로 지정해 프리셋 자세 이동을 요청합니다.
- `/packee/arm/pick_product` (`shopee_interfaces/srv/ArmPickProduct`)  
  Vision 모듈이 전달하는 목표 자세 정보를 바탕으로 픽업 시퀀스를 수행합니다.
- `/packee/arm/place_product` (`shopee_interfaces/srv/ArmPlaceProduct`)  
  보유 중인 상품을 지정 위치에 안전하게 내려놓습니다.
- `/packee/arm/pose_status`, `/packee/arm/pick_status`, `/packee/arm/place_status`  
  진행 단계와 메시지를 Packee Main으로 보고합니다.

## 런치 사용법
```bash
colcon build --packages-select packee_arm --base-paths shopee_ros2/src
source install/setup.bash
ros2 launch packee_arm packee_arm.launch.py
```

### 주요 런치 인자
- `run_pymycobot_dual` (기본: true)  
  듀얼 팔 노드를 실행합니다. 이 값이 `true`일 때는 단일 팔 노드를 함께 실행하지 마세요.
- `pymycobot_enabled_arms` (기본: `'left,right'`)  
  듀얼 팔 노드가 제어할 팔 목록입니다. 필요 시 `'left'` 또는 `'right'` 로 제한할 수 있습니다.
- `run_pymycobot_left`, `run_pymycobot_right` (기본: false)  
  듀얼 팔 모드를 사용하지 않을 때 개별 팔 노드를 실행합니다. 두 값을 동시에 `true`로 설정하면 서비스 이름이 충돌하므로 한쪽만 활성화하세요.
- `left_serial_port`, `right_serial_port`  
  좌·우 myCobot에 연결된 USB 포트 경로입니다. 단일 팔만 사용하는 경우 불필요한 포트를 비워두세요.
- `pymycobot_move_speed`, `pymycobot_approach_offset`, `pymycobot_lift_offset`  
  좌표 명령 속도와 픽업/담기 시 접근·상승 오프셋을 조정합니다.
- `preset_pose_cart_view`, `preset_pose_standby`  
  프리셋 자세를 `[x, y, z, rx, ry, rz]` 배열(회전은 라디안)로 지정할 수 있습니다.
- `run_jetcobot_bridge`, `jetcobot_command_period`, `jetcobot_workspace_*`  
  JetCobot 브릿지 노드 실행 여부와 적분 주기, 작업 공간 제한을 설정합니다.

## 하드웨어 준비
- Elephant Robotics **myCobot 280** 듀얼 암을 기준으로 파라미터를 설정했습니다.
- Jetson 등 제어 장치에 `pymycobot` 패키지를 설치해야 합니다.
  ```bash
  pip3 install pymycobot
  ```
- 기본 작업 공간은 수평 반경 0.28 m, Z 범위 0.05~0.30 m로 제한하며 JetCobot 브릿지가 자동 보정합니다.

## 빌드 및 실행
```bash
cd <workspace>
colcon build --packages-select packee_arm --base-paths shopee_ros2/src
source install/setup.bash
```

- **듀얼 팔 제어(기본)**
  ```bash
  ros2 launch packee_arm packee_arm.launch.py
  ```
- **단일 팔만 사용**
  ```bash
  ros2 launch packee_arm packee_arm.launch.py \
    run_pymycobot_dual:=false run_pymycobot_left:=true
  ```
- **JetCobot 브릿지 비활성화 예시**
  ```bash
  ros2 launch packee_arm packee_arm.launch.py run_jetcobot_bridge:=false
  ```

## 개발 메모
- 파라미터 값은 런치 인자 또는 ROS 2 파라미터 서버를 통해 런타임에 조정할 수 있습니다.
- 좌·우 팔 노드는 동일한 코드 베이스를 공유하므로 기능 추가 시 양쪽에 동일하게 적용해야 합니다.
- Packee Main과의 정합성 유지를 위해 새로운 서비스 필드가 추가될 경우 `docs/InterfaceSpecification`을 함께 업데이트하세요.
