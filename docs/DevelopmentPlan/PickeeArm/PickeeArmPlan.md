# Pickee Arm 개발 계획

## 1. 개요

`pickee_arm` 패키지는 Pickee 로봇의 팔(Arm) 제어를 담당하는 ROS2 패키지입니다.

주요 역할은 `pickee_main`으로부터 목표 상품의 위치를 받아, 뎁스 카메라를 이용한 비주얼 서보잉(Visual Servoing) 기술로 상품을 정확하게 집고(Pick) 지정된 위치(카트)에 놓는(Place) 것입니다. `pickee_vision`으로부터 실시간 객체 위치 정보를 받아 정밀한 제어를 수행하며, MoveIt2와 같은 외부 모션 플래닝 라이브러리를 사용하지 않고 자체적으로 제어 알고리즘을 구현합니다.

## 2. 개발 목표

- **비주얼 서보잉 기반 정밀 제어**: 뎁스 카메라 정보를 활용하여 목표 상품까지의 도달 및 파지(Grasping)를 위한 실시간 로봇 팔 제어 알고리즘 구현
- **안정적인 상품 이송**: 지정된 위치(선반)에서 상품을 집어 로봇의 카트로 옮기는 작업의 안정성 및 성공률 확보
- **`pickee_main`과의 연동**: `pickee_main`의 상태 머신과 원활하게 연동되도록 ROS2 Action 인터페이스를 통한 작업 요청 및 상태 보고 기능 구현
- **안전 기능**: 작업 중 발생할 수 있는 충돌 및 예외 상황을 감지하고 안전하게 작업을 중단하는 기능 구현

## 3. 주요 기능

- **PickeeArmTask Action 서버**:
  - `pickee_main`으로부터 'PICK', 'PLACE', 'HOME' 등의 작업을 Action Goal로 수신합니다.
  - 작업 수행 중 현재 상태(e.g., `MOVING_TO_ITEM`, `GRASPING`, `MOVING_TO_CART`)를 Feedback으로 전송합니다.
  - 작업 완료 또는 실패 시 최종 결과를 Result로 전송합니다.
- **비주얼 서보잉(Visual Servoing) 제어기**:
  - 목표 상품의 3D 좌표를 입력받아, 뎁스 카메라 영상에서 해당 상품을 기준으로 End-Effector를 정밀하게 이동시킵니다.
  - Image-Based Visual Servoing (IBVS) 또는 Position-Based Visual Servoing (PBVS) 방식을 적용하여 구현합니다.
- **그리퍼(Gripper) 제어**:
  - 상품을 집거나 놓기 위한 그리퍼의 열기/닫기 기능을 제어합니다.
- **상태 보고**:
  - 현재 로봇 팔의 관절 상태, End-Effector 위치, 작업 상태 등을 주기적으로 퍼블리시합니다.

## 4. 개발 및 테스트 환경

- **언어**: C++ / Python
- **ROS 버전**: ROS2 Jazzy
- **주요 라이브러리**:
  - `OpenCV`: 이미지 처리 및 비주얼 서보잉 알고리즘 구현
  - `PCL (Point Cloud Library)`: 뎁스 카메라의 포인트 클라우드 데이터 처리 (필요시)
  - `rclcpp` / `rclpy`: ROS2 통신
- **시뮬레이션 환경**: Gazebo (센서 및 로봇 모델 연동)
- **실제 하드웨어**: Pickee 로봇 Arm, 뎁스 카메라 (e.g., RealSense)

## 5. ROS2 인터페이스

- **Action Server**:
  - `~/pick_item` (`shopee_interfaces/action/PickeeArmTask`)
    - `pickee_main`으로부터 상품 피킹/플레이스 작업을 요청받습니다.
    - **Goal**: 작업 타입 (`PICK`/`PLACE`), 목표 상품 정보 (`product_id`, `pose`)
    - **Feedback**: 현재 작업 단계 (`status_code`)
    - **Result**: 작업 성공 여부 (`success`)
- **Publisher**:
  - `~/status` (`shopee_interfaces/msg/ArmPoseStatus`)
    - 현재 로봇 팔의 관절 각도, End-effector 좌표 등 상태를 게시합니다.
- **Subscriber**:
  - (필요시) `pickee_vision`으로부터 직접 객체 정보를 받을 경우 해당 토픽 구독
    - `/pickee_vision/detected_product` (`shopee_interfaces/msg/PickeeDetectedProduct`)
    - *참고: `InterfaceSpecification/Pic_Main_vs_Pic_Arm.md`에 따라, `pickee_main`이 `vision` 정보를 취합하여 `pickee_arm`에 Action Goal로 전달하는 것이 원칙이므로, 직접 구독은 필요하지 않을 수 있습니다.*

## 6. 개발 계획 (Milestones)

| Milestone | 주요 개발 내용                                         | 예상 기간 | 결과물                                       |
| :-------- | :----------------------------------------------------- | :-------- | :------------------------------------------- |
| M1        | **기본 패키지 및 인터페이스 설정**<br/>- ROS2 패키지 구조 생성<br/>- `PickeeArmTask` Action 서버/클라이언트 기본 코드 작성 | 1주       | Action 통신이 가능한 기본 노드               |
| M2        | **비주얼 서보잉 핵심 로직 구현**<br/>- 뎁스 카메라 연동 및 이미지/포인트클라우드 수신<br/>- 목표 좌표 기반 Arm 제어 알고리즘 초안 구현 | 2주       | 목표 지점으로 Arm을 이동시키는 기본 기능     |
| M3        | **상품 피킹(Picking) 기능 구현**<br/>- 그리퍼 제어 기능 추가<br/>- 비주얼 서보잉을 이용한 정밀 파지 로직 구현 | 2주       | 지정된 상품을 집는 기능 완료                 |
| M4        | **상품 플레이스(Placing) 기능 구현**<br/>- 카트의 지정된 위치로 Arm을 이동시키는 로직 구현<br/>- 그리퍼를 열어 상품을 놓는 기능 구현 | 1주       | 집은 상품을 카트에 내려놓는 기능 완료        |
| M5        | **통합 및 안정화**<br/>- `pickee_main`과의 Action 인터페이스 연동 테스트<br/>- 시나리오(SC-02-2) 기반 전체 프로세스 테스트 및 예외 처리 | 2주       | 안정적으로 피킹/플레이스가 가능한 `pickee_arm` 패키지 |

## 7. 테스트 계획

- **단위 테스트**:
  - 비주얼 서보잉 제어 알고리즘의 정확성 검증
  - 그리퍼 제어 함수의 동작 여부 검증
- **통합 테스트**:
  - Mock 노드를 사용하여 `pickee_main`과의 Action 통신 및 데이터 형식 검증
  - `pickee_vision`의 Mock Publisher를 이용해 가상 상품 위치를 제공하고, `pickee_arm`이 정상 동작하는지 검증
- **시나리오 테스트**:
  - **(시뮬레이션)** Gazebo 환경에서 `pickee_main`, `pickee_vision`, `pickee_arm`을 모두 실행하여 상품 피킹 시나리오(SC-02-2) 전체를 테스트
  - **(실제 환경)** 실제 Pickee 로봇을 이용하여 선반의 상품을 카트로 옮기는 전체 워크플로우의 성공률 및 소요 시간 측정
