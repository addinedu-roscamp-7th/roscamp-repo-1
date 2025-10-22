# Pickee Mobile Controller (C++ 구현)

Shopee Pickee Mobile의 C++ 기반 ROS2 구현입니다.

## 개요

이 패키지는 Pickee Mobile 로봇의 핵심 제어 로직을 Modern C++17로 구현한 것입니다.
고성능, 안전성, 확장성을 고려하여 설계되었습니다.

## 주요 기능

- **상태 기계 기반 제어**: IDLE, MOVING, STOPPED, CHARGING, ERROR 상태 관리
- **고성능 위치 추정**: Eigen 라이브러리 기반 센서 융합
- **실시간 경로 계획**: A* 및 DWA 알고리즘 지원
- **안전한 모션 제어**: PID 제어 및 비상 정지 기능
- **Modern C++ 설계**: 스마트 포인터, RAII, Exception Safety 적용

## 빌드 방법

```bash
# 의존성 설치
sudo apt install libeigen3-dev

# 빌드
cd /home/wonho/tech_research/Shopee/shopee_ros2
colcon build --packages-select pickee_mobile_wonho

# 환경 설정
source install/setup.bash
```

## 실행 방법

```bash
# 기본 실행
ros2 launch pickee_mobile_wonho pickee_mobile_controller.launch.py

# 파라미터 지정 실행
ros2 launch pickee_mobile_wonho pickee_mobile_controller.launch.py robot_id:=2
```

## 테스트 실행

```bash
# 모든 테스트 실행
colcon test --packages-select pickee_mobile_wonho

# 테스트 결과 확인
colcon test-result --verbose
```

## 아키텍처

### 주요 컴포넌트

1. **StateMachine**: 상태 전환 관리
2. **LocalizationComponent**: 위치 추정
3. **PathPlanningComponent**: 경로 계획
4. **MotionControlComponent**: 모션 제어
5. **SensorManager**: 센서 데이터 관리
6. **BatteryManager**: 배터리 관리
7. **CommunicationInterface**: 외부 통신

### 상태 다이어그램

```
IDLE ←→ MOVING ←→ STOPPED
 ↑        ↓         ↓
 ↑    CHARGING     ↓
 ↑                 ↓
 ← ← ← ERROR ← ← ← ←
```

## 토픽 인터페이스

### 구독 토픽
- `/scan` (sensor_msgs/LaserScan): LiDAR 데이터
- `/imu` (sensor_msgs/Imu): IMU 데이터  
- `/odom` (nav_msgs/Odometry): 오도메트리 데이터

### 발행 토픽
- `/cmd_vel` (geometry_msgs/Twist): 속도 명령
- `/battery_status` (std_msgs/Float64): 배터리 상태

## 파라미터

주요 파라미터는 `config/pickee_mobile_params.yaml`에서 설정 가능합니다:

- `robot_id`: 로봇 고유 ID
- `default_linear_speed`: 기본 선속도 (m/s)
- `default_angular_speed`: 기본 각속도 (rad/s)
- `battery_threshold_low`: 배터리 부족 임계값 (%)

## 개발 가이드

### 코딩 표준

- Modern C++17 사용
- 스마트 포인터 활용 (`std::unique_ptr`, `std::shared_ptr`)
- RAII 패턴 준수
- Exception Safety 보장

### 새로운 상태 추가

1. `include/pickee_mobile_wonho/states/` 에 헤더 파일 생성
2. `State` 클래스 상속
3. `OnEnter()`, `Execute()`, `OnExit()` 구현
4. `StateType` 열거형에 새로운 타입 추가

## 성능 최적화

- Eigen 라이브러리를 활용한 SIMD 최적화
- Lock-free 프로그래밍 적용
- 메모리 풀링을 통한 메모리 관리 최적화
- 컴파일러 최적화 옵션 활용 (-O3, -march=native)

## 라이선스

MIT License