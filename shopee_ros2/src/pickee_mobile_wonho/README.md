# Pickee Mobile Controller (C++ 구현)

이 패키지는 Shopee 로봇 쇼핑 시스템의 Pickee 로봇을 위한 이동 제어 노드입니다.

## 개요

`pickee_mobile_wonho` 패키지는 C++로 구현된 ROS2 패키지로, Pickee 로봇의 이동 제어를 담당합니다.

## 기능

- **이동 제어**: 전진, 후진, 좌회전, 우회전, 정지 명령 처리
- **오도메트리**: 로봇의 위치와 속도 정보 발행
- **센서 처리**: 레이저 스캐너 데이터를 이용한 장애물 감지
- **TF 발행**: 좌표계 변환 정보 제공
- **비상 정지**: 응급 상황 시 즉시 정지 기능

## 노드 구성

### pickee_mobile_controller

메인 제어 노드로서 다음과 같은 인터페이스를 제공합니다:

#### Services
- `pickee_mobile_move_to_location` (`shopee_interfaces/srv/PickeeMobileMoveToLocation`): 위치 이동 명령 서비스

#### Publishers
- `odom` (`nav_msgs/msg/Odometry`): 오도메트리 정보
- `cmd_vel` (`geometry_msgs/msg/Twist`): 속도 명령
- `mobile_status` (`std_msgs/msg/String`): 로봇 상태 정보

#### Subscribers
- `scan` (`sensor_msgs/msg/LaserScan`): 레이저 스캔 데이터
- `cmd_vel_input` (`geometry_msgs/msg/Twist`): 외부 속도 명령
- `pickee_mobile_speed_control` (`shopee_interfaces/msg/PickeeMobileSpeedControl`): 속도 제어 명령

#### Parameters
- `robot_id` (int, default: 1): 로봇 식별자
- `base_frame_id` (string, default: "base_link"): 로봇 베이스 프레임 ID
- `odom_frame_id` (string, default: "odom"): 오도메트리 프레임 ID
- `wheel_radius` (double, default: 0.035): 바퀴 반지름 [m]
- `wheel_base` (double, default: 0.160): 바퀴 간 거리 [m]
- `max_linear_velocity` (double, default: 0.5): 최대 직선 속도 [m/s]
- `max_angular_velocity` (double, default: 1.0): 최대 각속도 [rad/s]

## 사용법

### 빌드
```bash
cd ~/ros2_ws
colcon build --packages-select pickee_mobile_wonho
source install/setup.bash
```

### 실행
```bash
ros2 run pickee_mobile_wonho pickee_mobile_controller
```

### 런치 파일 사용
```bash
ros2 launch pickee_mobile_wonho mobile_bringup.launch.xml
```

## 명령어 예시

### 위치 이동 명령
```bash
# 특정 위치로 이동
ros2 service call /pickee_mobile_move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "{robot_id: 1, order_id: 123, location_id: 5, target_pose: {x: 2.0, y: 3.0, theta: 1.57}}"
```

### 속도 제어 명령
```bash
# 전진
ros2 topic pub /pickee_mobile_speed_control shopee_interfaces/msg/PickeeMobileSpeedControl "{robot_id: 1, order_id: 123, speed_mode: 'forward', target_speed: 0.3, obstacles: [], reason: '목표지점으로 이동'}" --once

# 정지
ros2 topic pub /pickee_mobile_speed_control shopee_interfaces/msg/PickeeMobileSpeedControl "{robot_id: 1, order_id: 123, speed_mode: 'stop', target_speed: 0.0, obstacles: [], reason: '목표 도달'}" --once

# 비상정지
ros2 topic pub /pickee_mobile_speed_control shopee_interfaces/msg/PickeeMobileSpeedControl "{robot_id: 1, order_id: 123, speed_mode: 'emergency_stop', target_speed: 0.0, obstacles: [], reason: '장애물 감지'}" --once
```

### 직접 속도 제어
```bash
ros2 topic pub /cmd_vel_input geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.0}}" --once
```

## 디렉토리 구조

```
pickee_mobile_wonho/
├── CMakeLists.txt
├── package.xml
├── README.md
├── include/
│   └── pickee_mobile_wonho/
│       └── pickee_mobile_controller.hpp
├── src/
│   └── pickee_mobile_controller.cpp
├── launch/
│   ├── mobile_bringup.launch.xml
│   ├── navigation_launch.xml
│   └── ...
├── params/
│   ├── nav2_params.yaml
│   └── ...
├── urdf/
│   ├── robot.urdf.xacro
│   └── ...
├── map/
├── models/
├── meshes/
├── worlds/
└── rviz/
```

## 의존성

- `rclcpp`: ROS2 C++ 클라이언트 라이브러리
- `shopee_interfaces`: Shopee 프로젝트 커스텀 메시지/서비스
- `tf2_ros`: TF2 변환 라이브러리
- `geometry_msgs`: 기하학적 메시지
- `nav_msgs`: 내비게이션 메시지
- `sensor_msgs`: 센서 메시지

## 개발자 정보

- **패키지**: pickee_mobile_wonho
- **버전**: 0.0.0
- **라이선스**: Apache-2.0
- **메인테이너**: wonho