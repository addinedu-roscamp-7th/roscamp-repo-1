# Pickee Mobile ROS2 패키지

`pickee_mobile` 패키지는 Shopee 로봇 쇼핑 시스템의 Pickee Mobile 로봇을 위한 ROS2 기반 제어 시스템을 구현합니다. Pickee Main Controller의 지시에 따라 로봇의 자율 이동, 위치 추정, 경로 계획 및 모션 제어를 담당하며, 로봇의 상태를 보고합니다.

## 1. 패키지 개요

이 패키지는 Pickee Mobile 로봇의 핵심 제어 로직을 포함하며, 다음과 같은 주요 컴포넌트들로 구성됩니다.

*   **`mobile_controller.py`**: Pickee Mobile 로봇의 메인 컨트롤러 노드입니다. 로봇의 자율 이동, 위치 추정, 경로 계획 및 모션 제어를 담당하며, `PickeeMobileMoveToLocation` 서비스를 통해 이동 명령을 수신하고, `NavigateToPose` 액션을 사용하여 로봇의 자율 이동을 제어합니다. 또한, `cmd_vel_modified` 토픽을 구독하여 로봇의 속도를 계산하고, `PickeeMobileArrival` 및 `PickeeMobilePose` 토픽을 발행하여 로봇의 도착 정보와 현재 위치 및 상태를 보고합니다.
*   **`state_machine.py`**: 로봇의 상태를 관리하고 상태 간의 전환을 처리하는 일반적인 상태 기계 프레임워크를 제공합니다.
*   **`states/` 디렉토리**: `State` 추상 클래스를 상속받아 Pickee Mobile 로봇의 각 구체적인 상태(IDLE, MOVING, STOPPED, CHARGING, ERROR)를 정의합니다.
*   **`launch/` 디렉토리**: ROS2 노드들을 실행하기 위한 launch 파일들을 포함합니다. 시뮬레이션 환경 설정, 로봇 모델 로드, 내비게이션 스택 실행 등 다양한 시나리오를 위한 launch 파일들이 있습니다.
*   **`map/` 디렉토리**: 로봇 내비게이션에 사용되는 지도 파일(예: `.yaml`, `.pgm`)들을 포함합니다.
*   **`meshes/` 디렉토리**: 로봇 및 환경 모델의 3D 메시 파일(예: `.stl`, `.dae`)들을 포함합니다.
*   **`models/` 디렉토리**: Gazebo 시뮬레이션에서 사용되는 3D 모델 정의 파일들을 포함합니다. (예: `desk`, `factory_L1`, `shelf`, `shopee_map` 등)
*   **`params/` 디렉토리**: ROS2 노드의 파라미터 설정 파일(예: `.yaml`)들을 포함합니다. 내비게이션 스택, SLAM 등에 필요한 설정들이 정의되어 있습니다.
*   **`rviz/` 디렉토리**: ROS 시각화 도구인 RViz의 설정 파일(예: `.rviz`)들을 포함합니다. 맵 뷰, 내비게이션 뷰, 로봇 모델 뷰 등을 위한 설정들이 있습니다.
*   **`urdf/` 디렉토리**: 로봇의 URDF(Unified Robot Description Format) 및 XACRO 파일들을 포함합니다. 로봇의 링크, 조인트, 센서 등을 정의합니다.
*   **`worlds/` 디렉토리**: Gazebo 시뮬레이션 환경을 정의하는 월드 파일(예: `.world`)들을 포함합니다.

## 2. ROS2 인터페이스

### 2.1. 토픽 (Topics)

| 토픽 이름                       | 메시지 타입                               | 설명                                                              | Pub/Sub   |
| :------------------------------ | :---------------------------------------- | :---------------------------------------------------------------- | :-------- |
| `/pickee/mobile/pose`           | `shopee_interfaces/msg/PickeeMobilePose`  | 로봇의 현재 위치, 속도, 배터리 잔량 및 상태를 보고                | Publish   |
| `/pickee/mobile/arrival`        | `shopee_interfaces/msg/PickeeMobileArrival` | 로봇이 목적지에 도착했음을 알림                                   | Publish   |
| `/cmd_vel_modified`             | `geometry_msgs/msg/Twist`                 | 로봇의 선속도 및 각속도 제어 명령 (하위 레벨 모션 제어)           | Subscribe |

### 2.2. 서비스 (Services)

| 서비스 이름                       | 서비스 타입                                       | 설명                                                              | Server/Client |
| :-------------------------------- | :------------------------------------------------ | :---------------------------------------------------------------- | :------------ |
| `/pickee/mobile/move_to_location` | `shopee_interfaces/srv/PickeeMobileMoveToLocation` | Pickee Main Controller로부터 특정 목적지로 이동 명령 수신         | Server        |

### 2.3. 액션 (Actions)

| 액션 이름                       | 액션 타입                                       | 설명                                                              | Client/Server |
| :------------------------------ | :------------------------------------------------ | :---------------------------------------------------------------- | :------------ |
| `/navigate_to_pose`             | `nav2_msgs/action/NavigateToPose`                 | Nav2 스택의 목적지 내비게이션 액션 (PickeeMobileController가 클라이언트로 사용) | Client        |


## 3. Mock 노드 (테스트 및 시뮬레이션용)

`pickee_mobile` 패키지에는 `mobile_controller` 노드와의 통신을 테스트하고 시뮬레이션하기 위한 여러 Mock 노드들이 포함되어 있습니다.

*   **`mock_speed_control_publisher.py`**: `/pickee/mobile/speed_control` 토픽을 **Publish**하여 `mobile_controller`의 속도 제어 기능을 테스트합니다.
*   **`mock_move_to_location_client.py`**: `/pickee/mobile/move_to_location` 서비스에 **Request**를 보내 `mobile_controller`의 이동 명령 처리 기능을 테스트합니다.
*   **`mock_update_global_path_client.py`**: `/pickee/mobile/update_global_path` 서비스에 **Request**를 보내 `mobile_controller`의 경로 업데이트 기능을 테스트합니다.
*   **`mock_pose_subscriber.py`**: `/pickee/mobile/pose` 토픽을 **Subscribe**하여 `mobile_controller`가 발행하는 로봇의 위치 정보를 모니터링합니다.
*   **`mock_arrival_and_move_status_subscriber.py`**: `/pickee/mobile/arrival` 토픽과 `/pickee/mobile/local_path` 토픽을 **Subscribe**하여 `mobile_controller`의 도착 알림 및 지역 경로 정보를 모니터링합니다.
*   **`goal_test/` 디렉토리**: 목표 설정 및 전송 기능을 테스트하기 위한 스크립트들을 포함합니다.
*   **`topic_test/` 디렉토리**: ROS2 토픽 통신을 테스트하기 위한 스크립트들을 포함합니다.

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

### 4.3. 프로젝트 노드 실행

**메인 컨트롤러 노드:**



```bash
#시뮬레이션
ros2 launch pickee_mobile gazebo_bringup.launch.xml 
ros2 launch pickee_mobile nav2_bringup_launch.xml use_sim_time:=True
ros2 launch pickee_mobile nav2_view.launch.xml
# 속도 제어 토픽 발행 (pub_cmd_vel 노드는 /cmd_vel_modified 토픽을 발행)
ros2 run pickee_mobile pub_cmd_vel 
```

```bash
#실제 로봇
ros2 launch pickee_mobile mobile_bringup.launch.xml #로봇
ros2 launch pickee_mobile nav2_bringup_launch.xml #로봇
ros2 run pickee_mobile pub_cmd_vel # 속도 제어 토픽 발행
```

기존에는 /cmd_vel 토픽을 발행해서 pickee의 속도를 제어 했는데 이제는 속도 제어를 위해 /cmd_vel_modified 토픽을 받아서 pickee의 속도를 제어 한다. 따라서 위의 3개의 launch 파일만 실행하면 로봇이 움직이지 않기 때문에 4번째 노드를 실행해야 한다.

**Mock 노드 (각각 별도의 터미널에서 실행):**
```bash
ros2 run pickee_mobile mock_speed_control_publisher
ros2 run pickee_mobile mock_move_to_location_client
ros2 run pickee_mobile mock_update_global_path_client
ros2 run pickee_mobile mock_pose_subscriber
ros2 run pickee_mobile mock_arrival_and_move_status_subscriber
```

## 5. 부분 기능 실행

이하의 코드들은 테스트에 사용해서 버전이 업데이트된 이후로는 불완전하거나 기능을 수행하지 못할 수 있다. 언급하지 않은 코드들은 그닥 중요하지 않은 코드이다.
기본적으로 위의 메인 컨트롤러 노드들을 실행한 이후에 rviz 창을 확인하며 사용해야 한다.

### 4.1. goal_test

지정 좌표 얻기, 목적지 지정, 현재 좌표 읽기를 하는 노드

*   **`send_goal_gui.py`**: gui를 사용해서 로봇을 특정 좌표, 각도로 이동시킨다.


### 4.2. topic_test

로봇의 속도, 위치, 목적지 등의 토픽을 `subscribe`하거나 `publish` 하는 노드

*   **`control_vel.py`**: `/cmd_vel`토픽을 **Subscribe**해서 속도 설정에 맞게 값을 변경한 후 `/cmd_vel_modified` 토픽으로 **Publish** 하는 노드, 현재 속도 변경과 일시정지는 작동하지 않는다.

*   **`pub_cmd_vel.py`**: `/cmd_vel` 토픽을 **Subscribe**하여 `/cmd_vel_modified` 토픽으로 **Publish**하는 노드. (주로 테스트 및 디버깅용)

*   **`pub_pose.py`**: 로봇의 `/cmd_vel`, `/amcl_pose` 등의 정보 토픽들을 **Subscribe** 해서 인터페이스 명세서에 맞게 **Publish**해주는 노드. 로봇이 이동중인지 도착(정지)중인지 추가해야 한다.
