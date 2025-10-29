# Pickee Mobile ROS2 패키지

`pickee_mobile` 패키지는 Shopee 로봇 쇼핑 시스템의 Pickee Mobile 로봇을 위한 ROS2 기반 제어 시스템을 구현합니다. Pickee Main Controller의 지시에 따라 로봇의 자율 이동, 위치 추정, 경로 계획 및 모션 제어, 속도제어를 담당하며, 로봇의 상태를 보고합니다.

## 1. 패키지 개요

이 패키지는 Pickee Mobile 로봇의 핵심 제어 로직을 포함하며, 다음과 같은 주요 컴포넌트들 과 테스트 컴포넌트로 구성됩니다.

main 폴더 : 핵심기능이 구현된 코드가 있는 폴더. 코드들은 모두 기능구현이 완료된 상태이다.
test 폴더 : 기능구현을 연습하면서 사용했던 테스트 코드가 있는 폴더. mock_test 폴더에는 통신 테스트를 위한 코드, goal_test 폴더는 목적지 지정 및 위치 토픽 구독을 테스트한 코드, topic_test 폴더에는 속도제어, 위티 토픽 발행 테스트 코드가 있다.

### 1.1. 특이사항

PickeeMobile가 받는 속도 토픽명은 **/cmd_vel_modified**이다. 속도제어, /cmd_vel 토픽 수정을 위해 PickeeMobile의 토픽 설정을 바꿨다. 속도 관련 오류가 있다면 이와 관련된 파일들을 확인하기 바란다. 

## 2. 실행하는법

### 2.1. 시뮬레이션
시뮬레이션을 안해봤는데 아마도 이렇게 하면 될거다.
```bash
#시뮬레이션
ros2 launch pickee_mobile gazebo_bringup.launch.xml # 가제보 실행
ros2 launch pickee_mobile nav2_bringup_launch.xml map:=map1021_modify.yaml # nav2 실행
ros2 launch pickee_mobile nav2_view.launch.xml # rviz 실행
ros2 launch pickee_mobile mobile_controller.launch.xml # 속도값

```

### 2.2 실제주행
```bash
#시뮬레이션
ros2 launch pickee_mobile mobile_bringup.launch.xml # 로봇
ros2 launch pickee_mobile nav2_bringup_launch.xml map:=map1021_modify.yaml #로봇, 맵 설정은 바꿔도 됨
ros2 launch pickee_mobile nav2_view.launch.xml #pc
ros2 launch pickee_mobile mobile_controller.launch.xml #pc

```
### 2.3 Aruco marker 추적
```base
ros2 launch pickee_mobile mobile_bringup.launch.xml # 로봇
ros2 launch pickee_mobile nav2_bringup_launch.xml # 로봇
ros2 run pickee_mobile mobile_aruco_pub_1 # pc
ros2 run pickee_mobile aruco_follow_1 # pc
```
mobile_aruco_pub_1 = z 누르면 pub 시작, x 누르면 pub 종료, 목적지 도착 토픽 subscribe를 동작 조건으로 하고 싶으면 관련 부분 주석 해제, 

mobile_aruco_pub_2 = 위와 동일, 실행하면 실시간 영상도 나옴

aruco_follow_1 = ArucoPose를 subscribe 하면 해당 위치로 이동, 각도는 안맞춤

aruco_follow_2 = 각도도 맞춰줌

aruco_follow_3 = 만드는중, 목표와 현재위치에 따라 속도, 이동거리 변경 예정

.

.

.

.

.

상세설명
mobile_bringup.launch.xml = PickeeMobile 시동걸기, 수업 자료에 있던거

nav2_bringup_launch.xml map:=map1021_modify.yaml = nav2 실행, 해당 파일의 default map을 설정해도 된다.

nav2_view.launch.xml = rviz 실행, 수업 자료에 있던거

mobile_controller.launch.xml = mobile_controller 노드, mobile_vel_modifier 노드 실행

    mobile_controller 기능 =  목적지 지정 service server, 목적지로 주행 명령 action client, 현재PickeeMobile정보 publish, 도착정보 publish

    mobile_vel_modifier 기능 = cmd_vel subscribe, 설정에 맞게 속도 변경, cmd_vel_modified publish, 주행속도 조절


!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!이하는 정리 안함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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
# 속도 토픽 발행
ros2 run pickee_mobile pub_cmd_vel 
```

```bash
#실제 로봇
ros2 launch pickee_mobile mobile_bringup.launch.xml #로봇
ros2 launch pickee_mobile nav2_bringup_launch.xml #로봇
ros2 launch pickee_mobile nav2_view.launch.xml #pc
ros2 run pickee_mobile pub_cmd_vel
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

*   **`pub_cmd_vel.py`**: `/cmd_vel`토픽을 **Subscribe**해서 그대로 `/cmd_vel_modified` 토픽으로 **Publish** 하는 노드. 

*   **`pub_pose.py`**: 로봇의 `/cmd_vel`, `/amcl_pose` 등의 정보 토픽들을 **Subscribe** 해서 인터페이스 명세서에 맞게 **Publish**해주는 노드. 로봇이 이동중인지 도착(정지)중인지 추가해야 한다.
