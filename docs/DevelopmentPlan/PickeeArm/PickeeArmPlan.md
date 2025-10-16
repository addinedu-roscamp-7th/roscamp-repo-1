# Pickee Arm 개발 계획

## 1. 개요

`pickee_arm` 패키지는 Pickee 로봇의 팔(Arm) 제어를 담당하는 ROS2 패키지입니다.

주요 역할은 `pickee_main`으로부터 목표 상품의 위치를 받아, 뎁스 카메라를 이용한 비주얼 서보잉(Visual Servoing) 기술로 상품을 정확하게 집고(Pick) 지정된 위치(카트)에 놓는(Place) 것입니다. `pickee_vision`으로부터 실시간 객체 위치 정보를 받아 정밀한 제어를 수행하여 제어 알고리즘을 구현합니다.

## 2. 개발 목표

- **비주얼 서보잉 기반 정밀 제어**: 뎁스 카메라 정보를 활용하여 목표 상품까지의 도달 및 파지(Grasping)를 위한 실시간 로봇 팔 제어 알고리즘 구현
- **안정적인 상품 이송**: 지정된 위치(선반)에서 상품을 집어 로봇의 카트로 옮기는 작업의 안정성 및 성공률 확보
- **`pickee_main`과의 연동**: `pickee_main`의 상태 머신과 원활하게 연동되도록 ROS2 인터페이스를 통한 작업 요청 및 상태 보고 기능 구현
- **안전 기능**: 작업 중 발생할 수 있는 충돌 및 예외 상황을 감지하고 안전하게 작업을 중단하는 기능 구현

## 3. 주요 기능

- **서비스 서버**: `pickee_main`으로부터 '자세 변경', '상품 피킹', '상품 놓기' 등의 작업을 Service 요청으로 수신합니다.
- **상태 퍼블리셔**: 작업 수행 중 현재 상태(e.g., `MOVING_TO_ITEM`, `GRASPING`)를 토픽으로 발행하여 `pickee_main`에 보고합니다.
- **비주얼 서보잉(Visual Servoing) 제어기**:
  - 목표 상품의 3D 좌표를 입력받아, 뎁스 카메라 영상에서 해당 상품을 기준으로 End-Effector를 정밀하게 이동시킵니다.
  - Image-Based Visual Servoing (IBVS) 또는 Position-Based Visual Servoing (PBVS) 방식을 적용하여 구현합니다.
- **그리퍼(Gripper) 제어**: 상품을 집거나 놓기 위한 그리퍼의 열기/닫기 기능을 제어합니다.

## 4. 개발 및 테스트 환경

- **언어**: C++ / Python
- **ROS 버전**: ROS2 Jazzy
- **주요 라이브러리**:
  - `OpenCV`: 이미지 처리 및 비주얼 서보잉 알고리즘 구현
  - `Arducam ToF SDK`: ToF 카메라 데이터 수신 및 처리를 위한 공식 라이브러리
  - `PCL (Point Cloud Library)`: ToF 카메라로부터 얻은 포인트 클라우드 데이터 처리
  - `rclcpp` / `rclpy`: ROS2 통신
- **실제 하드웨어**: Pickee 로봇 Arm, ToF(Time-of-Flight) 뎁스 카메라 (e.g., Arducam)

## 5. ROS2 인터페이스

- **Service Server**:
 - /pickee/arm/move_to_pose (shopee_interfaces/srv/PickeeArmMoveToPose): pickee_main으로부터 특정
   자세(선반 보기, 카트 보기, 대기)로 이동하라는 요청을 받습니다.
   - Request: robot_id, order_id, pose_type
   - Response: success, message
 - /pickee/arm/pick_product (shopee_interfaces/srv/PickeeArmPickProduct): pickee_main으로부터 특정
   상품을 집으라는 요청을 받습니다.
   - Request: robot_id, order_id, target_product
   - Response: accepted, message
 - /pickee/arm/place_product (shopee_interfaces/srv/PickeeArmPlaceProduct): pickee_main으로부터
   집은 상품을 카트에 담으라는 요청을 받습니다.
   - Request: robot_id, order_id, product_id
   - Response: accepted, message

- **Publisher**:
 - /pickee/arm/pose_status (shopee_interfaces/msg/ArmPoseStatus): 자세 변경 작업의 진행 상태(진행
   중, 완료, 실패)를 pickee_main에 보고합니다.
 - /pickee/arm/pick_status (shopee_interfaces/msg/PickeeArmTaskStatus): 상품 피킹 작업의 상세 진행
   상태(경로 계획, 접근, 그립 등)를 pickee_main에 보고합니다.
 - /pickee/arm/place_status (shopee_interfaces/msg/PickeeArmTaskStatus): 상품 담기 작업의 상세
   진행 상태(이동, 놓기, 그립 해제 등)를 pickee_main에 보고합니다.

- **Subscriber**:
 - 이 명세에 따르면 pickee_arm은 pickee_main으로부터 Service 요청을 받아 작업을 수행하므로, 직접
   구독하는 토픽은 없습니다.


## 6. 개발 계획 (Milestones)

| Milestone | 주요 개발 내용                                         | 예상 기간 | 결과물                                       |
| :-------- | :----------------------------------------------------- | :-------- | :------------------------------------------- |
| M1        | **기본 패키지 및 인터페이스 설정**<br/>- ROS2 패키지 구조 생성<br/>- 서비스 서버/클라이언트 기본 코드 작성 | 1주       | Service 통신이 가능한 기본 노드               |
| M2        | **비주얼 서보잉 핵심 로직 구현**<br/>- 뎁스 카메라 연동 및 이미지/포인트클라우드 수신<br/>- 목표 좌표 기반 Arm 제어 알고리즘 초안 구현 | 2주       | 목표 지점으로 Arm을 이동시키는 기본 기능     |
| M3        | **상품 피킹(Picking) 기능 구현**<br/>- 그리퍼 제어 기능 추가<br/>- 비주얼 서보잉을 이용한 정밀 파지 로직 구현 | 2주       | 지정된 상품을 집는 기능 완료                 |
| M4        | **상품 플레이스(Placing) 기능 구현**<br/>- 카트의 지정된 위치로 Arm을 이동시키는 로직 구현<br/>- 그리퍼를 열어 상품을 놓는 기능 구현 | 1주       | 집은 상품을 카트에 내려놓는 기능 완료        |
| M5        | **통합 및 안정화**<br/>- `pickee_main`과의 Service 인터페이스 연동 테스트<br/>- 시나리오(SC_02_3, SC_02_4) 기반 전체 프로세스 테스트 및 예외 처리 | 2주       | 안정적으로 피킹/플레이스가 가능한 `pickee_arm` 패키지 |

## 7. 테스트 계획

`pickee_arm`의 안정성을 보장하기 위해 단위, 통합, 시나리오의 세 단계로 나누어 테스트를 진행합니다.

### 7.1. 단위 테스트 (Unit Tests)
- **기구학/제어 로직 검증**: 로봇 팔의 움직임을 계산하는 핵심 수학 함수(정기구학, 역기구학, 자코비안)들의 정확성을 검증합니다.
- **비주얼 서보잉 알고리즘 검증**: 시각 정보를 바탕으로 팔의 움직임을 제어하는 핵심 알고리즘의 정확성을 검증합니다.
- **상태 관리 및 데이터 변환 로직 검증**: 내부 동작 상태가 외부로 전달되는 ROS 메시지로 정확히 변환되는지 확인합니다.

### 7.2. 통합 테스트 (Integration Tests)
- **서비스 인터페이스 검증**: Mock 노드를 사용하여 `move_to_pose`, `pick_product`, `place_product` 각 서비스 요청 시, 명세에 맞는 응답과 함께 관련 상태 정보(`pose_status`, `pick_status`, `place_status`)가 토픽으로 정확히 발행되는지 검증합니다.
- **실패 상황 대응 검증**: 의도적으로 실패 가능한 상황(e.g., 집을 수 없는 목표 지정)을 만들고, 시스템이 `failed` 상태를 정확히 보고하는지 확인합니다.

### 7.3. 시나리오 테스트 (Scenario / E2E Tests)
- **실제 환경 테스트**: 실제 Pickee 로봇을 이용하여, `SC_02_3`과 `SC_02_4`의 전체 흐름(자세 변경 -> 피킹 -> 플레이스 -> 대기 자세 복귀)이 정상적으로 동작하는지 검증하고, 성공률 및 평균 작업 시간을 측정합니다.