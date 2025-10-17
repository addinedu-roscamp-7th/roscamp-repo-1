# Pickee Mobile ROS2 패키지

`pickee_mobile` 패키지는 Shopee 로봇 쇼핑 시스템의 Pickee Mobile 로봇을 위한 ROS2 기반 제어 시스템을 구현합니다. Pickee Main Controller의 지시에 따라 로봇의 자율 이동, 위치 추정, 경로 계획 및 모션 제어를 담당하며, 로봇의 상태를 보고합니다.

## 1. 패키지 개요

이 패키지는 Pickee Mobile 로봇의 핵심 제어 로직을 포함하며, 다음과 같은 주요 컴포넌트들로 구성됩니다.

*   **`mobile_controller.py`**: Pickee Mobile 로봇의 메인 컨트롤러 노드입니다. 로봇의 전반적인 동작을 관리하며, 상태 기계와 여러 컴포넌트(위치 추정, 경로 계획, 모션 제어)를 통합하고 조정합니다.
*   **`localization_component.py`**: 로봇의 위치를 추정하고, 추정된 위치 정보와 배터리 잔량, 현재 로봇 상태를 `/pickee/mobile/pose` 토픽으로 발행합니다.
*   **`path_planning_component.py`**: 로봇의 경로를 계획하는 컴포넌트입니다. Pickee Main Controller로부터 이동 명령 및 전역 경로를 수신하고, 이를 기반으로 지역 경로 정보를 계산하여 발행합니다.
*   **`motion_control_component.py`**: 로봇의 실제 움직임을 제어하는 컴포넌트입니다. 경로 계획 컴포넌트로부터 지역 경로 정보를 수신하고, 속도 제어 명령을 처리하여 로봇의 선속도 및 각속도를 계산하여 `/cmd_vel` 토픽으로 발행합니다.
*   **`state_machine.py`**: 로봇의 상태를 관리하고 상태 간의 전환을 처리하는 일반적인 상태 기계 프레임워크를 제공합니다.
*   **`states/` 디렉토리**: `State` 추상 클래스를 상속받아 Pickee Mobile 로봇의 각 구체적인 상태(IDLE, MOVING, STOPPED, CHARGING, ERROR)를 정의합니다.

## 2. ROS2 인터페이스

### 2.1. 토픽 (Topics)

| 토픽 이름                       | 메시지 타입                               | 설명                                                              | Pub/Sub   |
| :------------------------------ | :---------------------------------------- | :---------------------------------------------------------------- | :-------- |
| `/pickee/mobile/pose`           | `shopee_interfaces/msg/PickeeMobilePose`  | 로봇의 현재 위치, 속도, 배터리 잔량 및 상태를 보고                | Publish   |
| `/pickee/mobile/arrival`        | `shopee_interfaces/msg/PickeeMobileArrival` | 로봇이 목적지에 도착했음을 알림                                   | Publish   |
| `/pickee/mobile/speed_control`  | `shopee_interfaces/msg/PickeeMobileSpeedControl` | Pickee Main Controller로부터 속도 제어 명령 수신                  | Subscribe |
| `/pickee/mobile/local_path`     | `shopee_interfaces/msg/PickeeMoveStatus`  | 경로 계획 컴포넌트에서 계산된 지역 경로 정보 (내부 통신)          | Publish   |
| `/cmd_vel`                      | `geometry_msgs/msg/Twist`                 | 로봇의 선속도 및 각속도 제어 명령 (하위 레벨 모션 제어)           | Publish   |

### 2.2. 서비스 (Services)

| 서비스 이름                       | 서비스 타입                                       | 설명                                                              | Server/Client |
| :-------------------------------- | :------------------------------------------------ | :---------------------------------------------------------------- | :------------ |
| `/pickee/mobile/move_to_location` | `shopee_interfaces/srv/PickeeMobileMoveToLocation` | Pickee Main Controller로부터 특정 목적지로 이동 명령 수신         | Server        |
| `/pickee/mobile/update_global_path` | `shopee_interfaces/srv/PickeeMobileUpdateGlobalPath` | Pickee Main Controller로부터 전역 경로 업데이트 명령 수신         | Server        |

## 3. Mock 노드 (테스트 및 시뮬레이션용)

`pickee_mobile` 패키지에는 `mobile_controller` 노드와의 통신을 테스트하고 시뮬레이션하기 위한 여러 Mock 노드들이 포함되어 있습니다.

*   **`mock_speed_control_publisher.py`**: `/pickee/mobile/speed_control` 토픽을 **Publish**하여 `mobile_controller`의 속도 제어 기능을 테스트합니다.
*   **`mock_move_to_location_client.py`**: `/pickee/mobile/move_to_location` 서비스에 **Request**를 보내 `mobile_controller`의 이동 명령 처리 기능을 테스트합니다.
*   **`mock_update_global_path_client.py`**: `/pickee/mobile/update_global_path` 서비스에 **Request**를 보내 `mobile_controller`의 경로 업데이트 기능을 테스트합니다.
*   **`mock_pose_subscriber.py`**: `/pickee/mobile/pose` 토픽을 **Subscribe**하여 `mobile_controller`가 발행하는 로봇의 위치 정보를 모니터링합니다.
*   **`mock_arrival_and_move_status_subscriber.py`**: `/pickee/mobile/arrival` 토픽과 `/pickee/mobile/local_path` 토픽을 **Subscribe**하여 `mobile_controller`의 도착 알림 및 지역 경로 정보를 모니터링합니다.

## 4. 실행 방법

### 4.1. 패키지 빌드

```bash
# shopee_interfaces 패키지 빌드 (메시지/서비스 정의)
colcon build --packages-select shopee_interfaces

# pickee_mobile 패키지 빌드
colcon build --packages-select pickee_mobile
```

### 4.2. ROS2 환경 설정

```bash
source install/setup.bash
```

### 4.3. 노드 실행

**메인 컨트롤러 노드:**
```bash
ros2 run pickee_mobile mobile_controller
```

**Mock 노드 (각각 별도의 터미널에서 실행):**
```bash
ros2 run pickee_mobile mock_speed_control_publisher
ros2 run pickee_mobile mock_move_to_location_client
ros2 run pickee_mobile mock_update_global_path_client
ros2 run pickee_mobile mock_pose_subscriber
ros2 run pickee_mobile mock_arrival_and_move_status_subscriber
```

## 5. 라이선스

이 패키지는 Apache-2.0 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하십시오.
