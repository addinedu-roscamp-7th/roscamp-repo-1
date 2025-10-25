# Shopee Packee Arm Controller 상세 설계

## 1. 개요 (Overview)

`packee_arm_controller`는 Packee 로봇의 하위 제어 노드로서, 상위 컴포넌트인 `packee_main_controller`와 `docs/InterfaceSpecification/Pac_Main_vs_Pac_Arm.md`에서 정의된 ROS2 인터페이스를 통해 연동된다. Packee Main이 생성한 포장 작업 계획에 따라 좌/우 로봇 팔을 동기화하여 상품을 픽업하고 포장 박스로 이송하며, 진행 상태를 지속적으로 상위 컴포넌트에 보고한다.

**주요 책임**
- Packee Main으로부터 수신한 자세 전환, 상품 픽업, 상품 담기 명령을 검증하고 실행
- 각 명령의 단계별 진행 상황을 토픽으로 발행하여 Packee Main의 상태 전이와 Shopee App 모니터링을 지원
- 두-stream CNN 기반 Visual Servo 제어와 하드웨어 계층(드라이버, 그리퍼)을 추상화하고 비정상 상황을 감지하여 상위 컴포넌트에 실패 알림 제공

## 2. 노드 아키텍처
- **노드 이름**: `packee_arm_controller`
- **실행 파일**: `lib/packee_arm/packee_arm_controller`
- **실행 방식**: `ros2 run packee_arm packee_arm_controller`
- **배치 위치**: Packee 로봇 Arm Control Device (`docs/Architecture/SWArchitecture.md` 참고)

## 3. 상위 컴포넌트 연동 (vs. Packee Main)
- Packee Main이 상태 머신을 기반으로 포장 워크플로우를 주도하며, Arm Controller는 하위 작업 수행자로 동작한다.
- Packee Main은 Arm Controller 서비스 호출을 통해 동작을 지시하고, Arm Controller의 상태 토픽을 기반으로 다음 상태 전환을 결정한다. (`docs/SequenceDiagram/SC_03_3.md`, `SC_03_4.md`)
- Arm Controller는 상위에서 정의한 작업 식별자(`robot_id`, `order_id`, `product_id`, `arm_side`)를 그대로 사용하여 추적성을 유지한다.

## 4. 내부 구성
```
PackeeArmController (rclcpp::Node)
 ├─ ServiceHandlers
 │   ├─ MoveToPoseHandler
 │   ├─ PickProductHandler
 │   └─ PlaceProductHandler
 ├─ ExecutionManager
 │   ├─ CommandQueueManager (좌/우 팔 큐)
 │   ├─ ArmExecutionTask (비동기 작업 스레드)
 │   └─ ProgressUpdater (토픽 발행기)
 ├─ VisualServoModule
 │   ├─ TwoStreamPoseRegressor (CNN 추론)
 │   ├─ FeatureFusionUnit
 │   └─ PController4DOF (x, y, z, yaw)
 ├─ HardwareInterface
 │   ├─ ArmDriverProxy (left/right, velocity 명령)
 │   └─ GripperController
 └─ Diagnostics
     ├─ HealthMonitor
     └─ OperationLogger
```
- VisualServoModule은 eye-in-hand 카메라 이미지 쌍(목표/현재)을 받아 두 개의 CNN 스트림으로 포즈를 추정하고, 4자유도 P 제어기를 통해 엔드이펙터 속도를 산출한다.
- `ExecutionManager`는 Packee Main이 다수의 명령을 순차 또는 부분 병렬로 내릴 때 충돌 없이 처리하도록 책임지며, VisualServoModule의 수렴 여부를 기반으로 상태를 갱신한다.

## 5. 인터페이스 상세

### 5.1 서비스 서버
| 이름 | 타입 | 목적 | 비고 |
| --- | --- | --- | --- |
| `/packee/arm/move_to_pose` | `ArmMoveToPose.srv` | 지정된 포즈(`cart_view`, `standby`)로 로봇 팔 자세 변경 | `pose_type` 검증, 진행률 업데이트 |
| `/packee/arm/pick_product` | `ArmPickProduct.srv` | 특정 팔이 상품을 픽업 | Vision 결과 기반 좌표 사용, `arm_side` 검증 |
| `/packee/arm/place_product` | `ArmPlaceProduct.srv` | 픽업된 상품을 포장 박스에 담기 | 픽업 완료 상태 확인 후 진행 |

### 5.2 토픽 퍼블리셔
| 이름 | 타입 | 내용 | 수신자 |
| --- | --- | --- | --- |
| `/packee/arm/pose_status` | `ArmPoseStatus.msg` | 자세 명령 진행률, 메시지 | Packee Main, Shopee App |
| `/packee/arm/pick_status` | `ArmTaskStatus.msg` | 픽업 단계(current_phase, progress) | Packee Main, Shopee App |
| `/packee/arm/place_status` | `ArmTaskStatus.msg` | 담기 단계(current_phase, progress) | Packee Main, Shopee App |

### 5.3 기타 인터페이스
- 향후 `/diagnostics` 토픽 연동을 통해 HealthMonitor 결과를 발행하여 Packee Main의 장애 대응 로직과 연계한다.

## 6. Visual Servo Controller 상세

### 6.1 제어 목표
- Eye-in-hand 카메라를 기반으로 엔드이펙터의 4자유도 `(x, y, z, yaw)`를 제어한다.
- 두-stream CNN이 출력한 목표 자세 `r* = (x*, y*, z*, Rz*)`와 현재 자세 `r_c = (x_c, y_c, z_c, Rz_c)`의 차이를 이용해 오차 `e = r* - r_c`를 만들고, 각 축을 독립적인 P 제어기로 제어한다.
- 제어 목표는 `‖e‖ → 0` (번역 ±3 mm, yaw ±3° 이내)이며, 15 제어 스텝 내 수렴을 SLA로 정의한다.
- CNN 출력 신뢰도가 0.75 미만이거나 목표가 시야에서 이탈할 경우 제어를 일시 중단하고 상위 컴포넌트에 재탐지를 요청한다.

### 6.2 Two-stream CNN 구조
- 입력: 목표 이미지(Image1)와 현재 이미지(Image2) 480×360 → 중앙 crop 후 224×224 리사이즈.
- 각 스트림은 5개의 Convolution + ReLU + Max Pooling 블록과 3개의 Fully Connected 층으로 구성된다. 두 스트림의 출력은 연결층에서 결합되어 `(x, y, z, Rz)`를 회귀한다.
- 보조 출력으로 이미지 평면 키포인트와 깊이를 회귀해 학습 시 안정성을 높이며, 불확실성(로그-분산)을 추가로 추정해 제어 게인을 조절한다.
- 학습: 총 400장 이미지(목표/현재 페어), Epoch 100, Batch 20, 학습률 0.0005, weight decay 0.001, Dropout 0.5, 손실은 MSE 기반.

### 6.3 P 제어 스킴
- 축별 제어 법칙은 `v_i = -λ_i (r_i* - r_{c,i})`이며 yaw는 `wrap(Rz* - Rz_c)`로 ±π 범위로 맞춘다.
- 기본 게인은 `λ = 0.03`이며, CNN 추정 불확실도에 따라 축별로 자동 조정한다.
- 속도 명령은 `ArmDriverProxy`를 통해 전송되고, 명령 클리핑(0.2 m/s, 15 deg/s)과 조인트 변환을 통해 실제 하드웨어로 전달된다.
- Jacobian 계산이나 카메라 보정 없이도 CNN이 2D→3D 매핑을 제공하므로 제어 파이프라인이 단순하다.

### 6.4 학습/실험 요약
- 검증 오차: X 3.72 mm, Y 3.58 mm, Z 4.02 mm, yaw 3.02°.
- 시뮬레이터/실기 혼합 환경에서 20개의 초기 자세를 테스트했으며 18건이 15 스텝(평균 1.36 s) 내 수렴했다.
- 금속 블록의 강한 반사나 조명 불균일로 CNN 신뢰도가 0.7 이하로 떨어질 경우 제어가 일시 중단되는 문제가 관찰되었으며, 데이터 확대와 조명 개선으로 보완 예정이다.

## 7. 명령 처리 시퀀스

### 7.1 자세 변경 (`move_to_pose`)
1. Packee Main이 `pose_type`(`cart_view`, `standby`)을 명시하여 서비스 호출한다.
2. Arm Controller는 요청 파라미터를 검증하고, 해당 자세의 기준 이미지와 예상 포즈를 로드한다.
3. VisualServoModule이 목표 이미지 대비 현재 이미지를 비교하여 `r*`, `r_c`를 산출하고 P 제어로 엔드이펙터를 이동시킨다.
4. 진행률은 오차 크기에 따라 0.0 → 0.5 → 1.0으로 갱신되며 `pose_status`로 발행된다.
5. 허용 오차 내로 수렴하면 `accepted=true`, 실패 또는 시야 손실 시 `accepted=false`와 오류 메시지를 반환한다.

### 7.2 상품 픽업 (`pick_product`)
1. Packee Main이 좌/우 팔, 목표 상품, 목표 이미지/포즈 보조정보를 포함해 서비스를 호출한다.
2. Arm Controller는 `arm_side`, 좌표 범위, 현재 Queue 상태를 점검하고 작업을 예약한다.
3. VisualServoModule이 목표 grasp 이미지/포즈를 기준으로 4 DOF를 제어하며, 오차가 임계값 아래로 떨어지면 그리퍼를 폐합해 픽업을 완료한다.
4. 진행 단계는 `servoing` → `grasping` → `lifting`으로 단순화되며, 각 단계마다 `pick_status`를 발행한다.
5. 성공 시 `accepted=true`와 함께 해당 상품을 “place 대기” 상태로 표시하고, CNN 신뢰도가 낮거나 하드웨어 오류가 발생하면 실패 상태를 보고한다.

### 7.3 상품 담기 (`place_product`)
1. Packee Main이 동일 상품 ID와 박스 목표 이미지를 전달하여 서비스를 호출한다.
2. Arm Controller는 픽업 완료 여부를 확인하고, 박스 내부 목표 자세를 로드한다.
3. VisualServoModule이 목표 대비 현재 이미지를 기반으로 박스 상단에서 yaw 정렬 및 위치 맞춤을 수행하고, 오차 허용 범위에 도달하면 그리퍼를 개방한다.
4. 진행 단계는 `servoing` → `placing` → `retreat`로 정의하며, 각 단계마다 `place_status`를 발행한다.
5. 완료 후 안전 거리로 복귀하고 큐에서 해당 상품을 제거한다. 실패 시 원인(시야 손실, CNN 신뢰도 저하, 하드웨어 오류)을 상태 메시지에 포함한다.

## 8. 상태 및 예외 처리
- 내부 상태는 `IDLE`, `MOVING`, `PICKING`, `PLACING`, `ERROR`로 관리하여 Packee Main의 상위 상태(`StateDiagram_Packee.md`)와 매핑한다.
- 주요 예외 케이스와 대응:
  - 파라미터 오류: 즉시 거부, 상태 토픽 `failed` 발행.
  - CNN 신뢰도 저하: `current_phase='servoing'`, 재탐지 요청 여부를 메시지에 포함.
  - 그리퍼 이상: `current_phase='grasping'` 또는 `placing`에서 실패, 센서 코드 로그 기록.
  - 하드웨어 응답 타임아웃: `accepted=false`, Packee Main이 재시도/중단 판단.
- 모든 예외는 Diagnostics 모듈이 에러 코드와 함께 로깅하여 Shopee App 알림 요건(UR_04/SR_11)에 대응한다.

## 9. 파라미터 (ROS2 Parameters)
| 파라미터 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `robot_id` | int | 1 | 제어 대상 로봇 ID |
| `arm_sides` | string | `left,right` | 활성화된 팔 목록 |
| `servo_gain_xy` | double | 0.02 | X/Y 축 P 제어 게인 (myCobot 280 암 길이 기준) |
| `servo_gain_z` | double | 0.018 | Z 축 P 제어 게인 |
| `servo_gain_yaw` | double | 0.04 | Yaw 축 P 제어 게인 |
| `cnn_confidence_threshold` | double | 0.75 | CNN 신뢰도 임계값 |
| `max_translation_speed` | double | 0.05 | 엔드이펙터 최대 병진 속도 (m/s) |
| `max_yaw_speed_deg` | double | 40.0 | 엔드이펙터 최대 yaw 속도 (deg/s) |
| `gripper_force_limit` | double | 12.0 | 그리퍼 힘 제한 (N) |
| `progress_publish_interval` | double | 0.15 | 상태 토픽 발행 주기 (초) |
| `command_timeout_sec` | double | 4.0 | 하드웨어 응답 타임아웃 |
| `preset_pose_cart_view` | double[4] | `[0.16, 0.0, 0.18, 0.0]` | myCobot 280 카트 점검 자세 (x, y, z, yaw_deg) |
| `preset_pose_standby` | double[4] | `[0.10, 0.0, 0.14, 0.0]` | 기본 대기 자세 (x, y, z, yaw_deg) |
- 파라미터는 런타임에 조정 가능하도록 설계하며, Packee Main 파라미터(`component_service_timeout`)와 정합성을 유지한다.

- myCobot 280 기준 좌표계는 베이스 중심을 원점(미터)으로 하며, Z 축은 작업대 기준 0.0 m에서 시작한다. `preset_pose_*` 파라미터는 듀얼 암 구성 시 좌/우 동일 값을 적용하고, 필요 시 개별 값으로 오버라이드할 수 있다.

## 10. 로깅 및 진단
- ROS logger를 활용하여 서비스 수신, 검증, 단계 전환, 예외 발생 시 INFO/ERROR 로그를 남긴다.
- `HealthMonitor`는 주기적으로 토크, 온도 등 상태 데이터를 수집해 추후 `/diagnostics` 토픽 통합을 지원한다.
- 운영 로그는 Packee Main의 장애 분석과 Shopee App 알림에 활용된다.

## 11. 테스트 전략
- **단위 테스트**: 서비스 입력 검증, 상태 전이, VisualServoModule P 제어 로직의 경계 조건을 gtest+rclcpp 기반으로 확인한다.
- **통합 테스트**: `ros2 run packee_arm mock_packee_main` 노드와 연동해 서비스/토픽 흐름 및 CNN 신뢰도 시나리오(정상/저하)를 재현한다.
- **시뮬레이션 테스트**: Gazebo/Isaac Sim 환경에서 eye-in-hand 카메라와 금속 블록 모델을 사용해 조명 변화, 초기 오프셋 다양화에 대한 제어 성능을 검증한다.
- **HIL 테스트**: Elephant Robotics myCobot 280 듀얼 암과 실제 카메라, 금속 블록을 이용해 20개 이상 시나리오를 반복 실행하며 수렴 시간, 최종 오차를 측정한다.
- 테스트 결과는 Packee Main Controller 테스트와 함께 QA 리뷰에 제출한다.

## 12. 향후 과제
- 학습 데이터 확대 및 조명 조건 다양화로 CNN 일반화 성능 향상.
- glare 대응을 위한 광원 제어, 편광 필터 적용 및 이미지 전처리 개선.
- 두 팔 간 협업 시 다중 목표 이미지를 처리할 수 있도록 VisualServoModule 확장.
- 상태 토픽에 CNN 신뢰도, 수렴 스텝 수 등 추가 진단 필드를 포함하는 방안 검토.
- LLM 기반 음성/텍스트 명령과의 통합 시나리오 대비 추가 서비스 정의.

본 상세 설계는 구현 진행 상황에 따라 업데이트되며, 변경 시 상위 컴포넌트 설계(특히 Packee Main Controller)와의 정합성을 우선 확인한다.
