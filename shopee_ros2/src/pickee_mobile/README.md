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
#실제
ros2 launch pickee_mobile mobile_bringup.launch.xml # 로봇
ros2 launch pickee_mobile nav2_bringup_launch.xml map:=map1021_modify.yaml #로봇, 맵 설정은 바꿔도 됨
ros2 launch pickee_mobile nav2_view.launch.xml #pc
ros2 launch pickee_mobile mobile_controller.launch.xml #pc

```

상세설명
mobile_bringup.launch.xml = PickeeMobile 시동걸기, 수업 자료에 있던거

nav2_bringup_launch.xml map:=map1021_modify.yaml = nav2 실행, 해당 파일의 default map을 설정해도 된다.

nav2_view.launch.xml = rviz 실행, 수업 자료에 있던거

mobile_controller.launch.xml = mobile_controller 노드, mobile_vel_modifier 노드 실행

    mobile_controller 기능 =  목적지 지정 service server, 목적지로 주행 명령 action client, 현재PickeeMobile정보 publish, 도착정보 publish

    mobile_vel_modifier 기능 = cmd_vel subscribe, 설정에 맞게 속도 변경, cmd_vel_modified publish, 주행속도 조절

    
### 2.3 Aruco marker 도킹
```base
ros2 launch pickee_mobile mobile_bringup.launch.xml # 로봇
ros2 launch pickee_mobile nav2_bringup_launch.xml # 로봇 맵 설정은 원하는대로
ros2 launch pickee_mobile nav2_view.launch.xml #pc
ros2 launch pickee_mobile mobile_controller # pc
ros2 run pickee_mobile main_aruco_pub # pc


TEST_GUIDE의 /pickee/mobile/move_to_location service_client 서비스 요청, 해당 위치로 이동 후 도킹함
```

상세설명
mobile_bringup.launch.xml = PickeeMobile 시동걸기, 수업 자료에 있던거

nav2_bringup_launch.xml map:=map1021_modify.yaml = nav2 실행, 해당 파일의 default map을 설정해도 된다.

nav2_view.launch.xml = rviz 실행, 수업 자료에 있던거

mobile_controller.launch.xml = mobile_controller 노드, mobile_vel_modifier 노드, mobile_aruco_docking 실행

mobile_aruco_docking = aruco 마커 위치 토픽 subscribe 해서 도킹 프로세스 실행, module_go_straite.py, module_rotate.py 함수 import, 내부 코드에서 각도는 전부 rad단위, 로그에 뜨는건 degree 단위

main_aruco_pub = aruco 마커 위치 토픽 publish, module_aruco_detect.py 함수 import

---
주의사항 모든 속도, 주행 관련 노드(docking, go_straight, rotate, controller)에서 속도 관련 토픽명을 /cmd_vel_modified로 설정했는데 사용자 설정에 맞게 수정해서 사용 바람, 아니면 mobile_vel_modifier 노드에서 subscribe 를 /cmd_vel_modified로 하고 publish를 /cmd_vel로 하면 주행 관련 노드들 수정은 안해도 될거임
---