# Shopee Packee Arm Controller 상세 설계

## 1. 개요 (Overview)

`packee_arm_controller`는 Packee 로봇의 하위 제어 노드로서, 상위 컴포넌트인 `packee_main_controller`와 `docs/InterfaceSpecification/Pac_Main_vs_Pac_Arm.md`에서 정의된 ROS2 인터페이스를 통해 연동된다. Packee Main이 생성한 포장 작업 계획에 따라 좌/우 로봇 팔을 동기화하여 상품을 픽업하고 포장 박스로 이송하며, 진행 상태를 지속적으로 상위 컴포넌트에 보고한다.

**주요 책임**
- Packee Main으로부터 수신한 자세 전환, 상품 픽업, 상품 담기 명령을 검증하고 실행
- 각 명령의 단계별 진행 상황을 토픽으로 발행하여 Packee Main의 상태 전이와 Shopee App 모니터링을 지원
- 하드웨어 제어 계층(모션 플래너, 드라이버, 그리퍼)과의 연동을 추상화하고 비정상 상황을 감지하여 상위 컴포넌트에 실패 알림 제공

## 2. 노드 아키텍처
- **노드 이름**: `packee_arm_controller`
- **실행 파일**: `lib/packee_arm/packee_arm_controller` (C++), `lib/packee_arm/arm_controller.py` (테스트용 rclpy 래퍼)
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
 ├─ HardwareInterface
 │   ├─ MotionPlannerAdapter
 │   ├─ ArmDriverProxy (left/right)
 │   └─ GripperController
 └─ Diagnostics
     ├─ HealthMonitor
     └─ OperationLogger
```
- 현재 구현은 단일 Python 파일 구조이나, 상기 모듈 분리를 기준으로 리팩터링을 진행한다.
- `ExecutionManager`는 Packee Main이 다수의 명령을 순차/부분 병렬로 내릴 때 충돌 없이 처리하도록 책임진다.

## 5. 인터페이스 상세

### 5.1 서비스 서버
| 이름 | 타입 | 목적 | 비고 |
| --- | --- | --- | --- |
| `/packee/arm/move_to_pose` | `PackeeArmMoveToPose.srv` | 지정된 포즈(`cart_view`, `standby`)로 로봇 팔 자세 변경 | `pose_type` 검증, 진행률 업데이트 |
| `/packee/arm/pick_product` | `PackeeArmPickProduct.srv` | 특정 팔이 상품을 픽업 | Vision 결과 기반 좌표 사용, `arm_side` 검증 |
| `/packee/arm/place_product` | `PackeeArmPlaceProduct.srv` | 픽업된 상품을 포장 박스에 담기 | 픽업 완료 상태 확인 후 진행 |

### 5.2 토픽 퍼블리셔
| 이름 | 타입 | 내용 | 수신자 |
| --- | --- | --- | --- |
| `/packee/arm/pose_status` | `ArmPoseStatus.msg` | 자세 명령 진행률, 메시지 | Packee Main, Shopee App |
| `/packee/arm/pick_status` | `PackeeArmTaskStatus.msg` | 픽업 단계(current_phase, progress) | Packee Main, Shopee App |
| `/packee/arm/place_status` | `PackeeArmTaskStatus.msg` | 담기 단계(current_phase, progress) | Packee Main, Shopee App |

### 5.3 기타 인터페이스
- 향후 `/diagnostics` 토픽 연동을 통해 HealthMonitor 결과를 발행하여 Packee Main의 장애 대응 로직과 연계한다.

## 6. 명령 처리 시퀀스

### 6.1 자세 변경 (`move_to_pose`)
1. Packee Main이 `pose_type`을 명시하여 서비스 호출 (`SC_03_3`, `SC_03_4` 1단계).
2. Arm Controller는 허용된 포즈 목록(`cart_view`, `standby`)을 검증하고, `ExecutionManager`를 통해 단일 동작으로 예약.
3. `MotionPlannerAdapter`가 경로를 생성하고, `ArmDriverProxy`가 실제 모션을 실행.
4. 진행률을 0.0 → 0.5 → 1.0으로 갱신하며 `pose_status`를 발행.
5. 성공 시 `accepted=true`, 실패 시 `accepted=false`와 오류 메시지로 응답.

### 6.2 상품 픽업 (`pick_product`)
1. Packee Main이 좌/우 팔과 상품 정보를 지정하여 서비스 호출.
2. Arm Controller는 `arm_side` 유효성, 좌표 범위, 현재 팔 작업 상태를 점검.
3. `ExecutionManager`가 단계별 태스크를 수행:
   - `planning`: 충돌 회피 경로 생성 (progress ≈ 0.2)
   - `approaching`: 목표 위치 접근 (progress ≈ 0.4)
   - `grasping`: 그리퍼 파지 및 센서 확인 (progress ≈ 0.6)
   - `lifting`: 안전 높이로 이동 (progress ≈ 0.8)
4. 각 단계에서 `pick_status`를 발행하며, 실패 시 즉시 `status=failed`.
5. 성공 시 `accepted=true`, 이후 `ExecutionManager`가 상품을 “담기 대기” 상태로 표시하여 Packee Main의 다음 명령을 대비한다.

### 6.3 상품 담기 (`place_product`)
1. Packee Main이 동일한 상품 ID와 팔 정보를 전달하여 서비스 호출.
2. Arm Controller는 해당 상품이 픽업 완료 상태인지 확인, 아니면 거부.
3. 단계별 진행:
   - `planning`: 박스 위치 기반 경로 생성
   - `approaching`: 박스로 접근
   - `placing`: 상품 투입 및 그리퍼 개방
   - `done`: 안전 자세 복귀
4. `place_status`를 통해 진행률을 보고하고, 성공 시 `accepted=true`.
5. 작업 완료 후 내부 큐에서 해당 상품을 제거하여 중복 지시를 방지한다.

## 7. 상태 및 예외 처리
- 내부 상태는 `IDLE`, `MOVING`, `PICKING`, `PLACING`, `ERROR`로 관리하여 Packee Main의 상위 상태(`StateDiagram_Packee.md`)와 매핑한다.
- 주요 예외 케이스와 대응:
  - 파라미터 오류: 즉시 거부, 상태 토픽 `failed` 발행.
  - 경로 계획 실패: `current_phase='planning'`, 재시도 가능 여부를 메시지에 포함.
  - 그리퍼 이상: `current_phase='grasping'` 또는 `placing`에서 실패, 센서 코드 로그 기록.
  - 하드웨어 응답 타임아웃: `accepted=false`, Packee Main이 재시도/중단 판단.
- 모든 예외는 Diagnostics 모듈이 에러 코드와 함께 로깅하여 Shopee App 알림 요건(UR_04/SR_11)에 대응한다.

## 8. 파라미터 (ROS2 Parameters)
| 파라미터 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `robot_id` | int | 1 | 제어 대상 로봇 ID |
| `arm_sides` | string | `left,right` | 활성화된 팔 목록 |
| `pose_speed_scale` | double | 0.6 | 자세 전환 속도 스케일 |
| `pick_speed_scale` | double | 0.4 | 픽업 시 속도 스케일 |
| `gripper_force_limit` | double | 35.0 | 그리퍼 힘 제한 (N) |
| `progress_publish_interval` | double | 0.2 | 상태 토픽 발행 주기 (초) |
| `command_timeout_sec` | double | 5.0 | 하드웨어 응답 타임아웃 |
- 파라미터는 런타임에 조정 가능하도록 설계하며, Packee Main 파라미터(`component_service_timeout`)와 정합성을 유지한다.

## 9. 로깅 및 진단
- ROS logger를 활용하여 서비스 수신, 검증, 단계 전환, 예외 발생 시 INFO/ERROR 로그를 남긴다.
- `HealthMonitor`는 주기적으로 토크, 온도 등 상태 데이터를 수집해 추후 `/diagnostics` 토픽 통합을 지원한다.
- 운영 로그는 Packee Main의 장애 분석과 Shopee App 알림에 활용된다.

## 10. 테스트 전략
- **단위 테스트**: 서비스 입력 검증, 진행률 계산, 상태 전이 함수에 대해 gtest+rclcpp 기반 테스트 작성.
- **통합 테스트**: `mock_packee_main.py`와 상호 작용하여 정상/실패 시나리오 반복 실행 (Sequence Diagram 재현).
- **시뮬레이션 테스트**: Gazebo/Isaac Sim에서 좌/우 팔 모델과 연동하여 충돌 회피, 동기화 검증.
- **HIL 테스트**: 실제 로봇 팔/그리퍼와 연결해 속도/힘 제한, 예외 처리, 복구 시나리오 검증.
- 테스트 결과는 Packee Main Controller 테스트와 함께 QA 리뷰에 제출한다.

## 11. 향후 과제
- MoveIt2 등 모션 플래너와의 실제 통합 구현 (`MotionPlannerAdapter` 보완).
- 듀얼 암 협업 최적화를 위한 작업 스케줄링 고도화 (Packee Main 계획과 연동).
- 상태 토픽 확장: 에러 코드, 재시도 횟수, 예상 완료 시간 등 추가 필드 검토.
- LLM 기반 음성/텍스트 명령과의 통합 시나리오 대비 추가 서비스 정의.

본 상세 설계는 구현 진행 상황에 따라 업데이트되며, 변경 시 상위 컴포넌트 설계(특히 Packee Main Controller)와의 정합성을 우선 확인한다.
