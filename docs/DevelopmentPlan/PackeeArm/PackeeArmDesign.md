# Shopee Packee Arm Controller 상세 설계

본 문서는 Python 기반 Packee Arm 제어 패키지(`packee_arm`)의 구조와 인터페이스를 정의한다.  
`docs/InterfaceSpecification/Pac_Main_vs_Pac_Arm.md`에서 규정한 서비스/토픽과 정합성을 유지하며, myCobot 280 하드웨어 제어는 `pymycobot` 라이브러리를 통해 수행한다.

## 1. 개요
- **주요 컴포넌트**
  - `pymycobot_dual` : 좌/우 팔을 동시에 제어하며, 공식 서비스·토픽 인터페이스를 단일 노드에서 처리한다.
  - `pymycobot_left` / `pymycobot_right` : 필요 시 단일 팔만 운용하기 위한 보조 노드.
  - `jetcobot_bridge.py` : 상위 모듈이 발행한 속도/그리퍼 명령을 myCobot 시리얼 명령으로 변환한다.
- **책임**
  - Packee Main이 호출하는 `/packee/arm/move_to_pose`, `/packee/arm/pick_product`, `/packee/arm/place_product` 서비스를 처리한다.
  - `/packee/arm/pose_status`, `/packee/arm/pick_status`, `/packee/arm/place_status` 토픽을 통해 진행 상황을 보고한다.
  - myCobot 좌표·그리퍼 명령을 직렬 포트로 전송하고, 예외 상황을 로깅한다.

## 2. 소프트웨어 아키텍처
```
Packee Main (ROS 2 services/topics)
        │
        ▼
┌───────────────────────────────────────────────┐
│ packee_arm 패키지 (Python)                    │
│                                               │
│  ┌────────────────────────────────────────┐
│  │ pymycobot_dual                         │
│  │  - 좌/우 Arm 서비스 처리               │
│  │  - 상태 토픽 발행                      │
│  │  - 좌표/그리퍼 명령 준비               │
│  └──────────┬─────────────────────────────┘
│             │
│  (단일 팔 모드 선택 시)                    │
│  ┌───────────────────────┐   ┌───────────────────────┐
│  │ pymycobot_left        │   │ pymycobot_right       │
│  │  - 좌측 Arm 서비스    │   │  - 우측 Arm 서비스    │
│  └──────────┬────────────┘   └──────────┬────────────┘
│             │                             │
│             ▼                             ▼
│        ┌────────────────────────────────────────┐
│        │ jetcobot_bridge.py                    │
│        │  - 좌우 팔 Twist/Float32 구독        │
│        │  - 속도 적분 → 좌표 변환             │
│        │  - pymycobot API 호출                │
│        └────────────────────────────────────────┘
```
- 기본 운용은 `pymycobot_dual`을 통해 좌/우 팔을 동시에 제어한다.
- 단일 팔만 필요한 경우 `run_pymycobot_dual:=false`로 설정하고, 원하는 팔 노드를 개별적으로 기동한다.
- 브릿지는 좌/우 팔 속도 명령을 적분해 좌표를 생성하고, `sync_send_coords`/`set_gripper_value` 명령을 전송한다.

## 3. 인터페이스 정합성
- **서비스**
  - `/packee/arm/move_to_pose` (`ArmMoveToPose.srv`)
  - `/packee/arm/pick_product` (`ArmPickProduct.srv`)
  - `/packee/arm/place_product` (`ArmPlaceProduct.srv`)
- **토픽**
  - `/packee/arm/pose_status` (`ArmPoseStatus.msg`)
  - `/packee/arm/pick_status` (`ArmTaskStatus.msg`)
  - `/packee/arm/place_status` (`ArmTaskStatus.msg`)

`pymycobot_dual` 내부에서 `arm_side` 값을 사용해 좌/우 팔을 구분한다. 단일 팔 모드에서는 기본 arm_side가 노드 이름에 맞춰 설정된다.

## 4. 주요 모듈 상세

### 4.1 `pymycobot_dual`
- `rclpy.node.Node`를 상속하며, 좌/우 팔을 동시에 관리한다.
- 주요 파라미터
  - `enabled_arms` (문자열, 기본 `'left,right'`)
  - `serial_port_left`, `serial_port_right`, `baud_rate`, `move_speed`
  - `approach_offset_m`, `lift_offset_m`
  - `preset_pose_cart_view`, `preset_pose_standby` (`[x, y, z, rx, ry, rz]`, 라디안)
  - `gripper_open_value`, `gripper_close_value`
  - 상태 토픽 경로 (`pose_status_topic`, `pick_status_topic`, `place_status_topic`)
- 서비스 처리 흐름
  1. 요청 검증(arm_side 유효성, 하드웨어 연결 여부)
  2. Pose6D → 내부 dict 변환 (`x`, `y`, `z`, `rx`, `ry`, `rz`)
  3. 접근/하강/그리퍼 시퀀스를 arm_side 별로 실행
  4. 진행률에 맞춰 상태 토픽을 발행
- `MoveToPose` 요청은 활성화된 모든 팔을 동일한 프리셋 자세로 이동시킨다.

### 4.2 `pymycobot_left` / `pymycobot_right`
- 듀얼 노드를 비활성화했을 때 사용할 수 있는 단일 팔 전용 노드.
- `serial_port`, `arm_sides` 등 핵심 파라미터는 듀얼 노드와 동일하며, 서비스/토픽 구현도 동일 로직으로 동작한다.

### 4.3 `jetcobot_bridge.py`
- 좌우 팔 Twist/Float32 명령을 구독해 myCobot 시리얼 명령으로 변환한다.
- Twist 속도를 적분해 `(x, y, z, rx, ry, rz)` 좌표를 누적하고, degree로 변환 후 `sync_send_coords` 호출.
- 작업 공간 제한
  - 반경: `workspace_radial`
  - Z 범위: `workspace_z_min` ~ `workspace_z_max`
- 파라미터
  - `left_serial_port`, `right_serial_port`
  - `command_period_sec`, `move_speed`
  - `default_pose_cart_view`, `default_pose_standby`
  - `gripper_open_value`, `gripper_close_value`

## 5. 런치 구성 (`launch/packee_arm.launch.py`)
- 런치 인자
  - `run_pymycobot_dual`, `run_pymycobot_left`, `run_pymycobot_right`, `run_jetcobot_bridge`
  - `pymycobot_enabled_arms`, `pymycobot_move_speed`, `pymycobot_approach_offset`, `pymycobot_lift_offset`
  - 상태 토픽 경로(`pose_status_topic`, `pick_status_topic`, `place_status_topic`)
  - 프리셋/그리퍼 설정, 시리얼 포트 및 브릿지 관련 인자
- 동작
  - 기본적으로 듀얼 팔 노드를 기동한다.
  - 단일 팔 모드를 사용하려면 `run_pymycobot_dual:=false`와 함께 원하는 팔 노드를 `true`로 설정한다.
  - 브릿지는 필요 시 함께 실행하며, 프리셋 자세 인자는 문자열/배열 모두 허용된다.

## 6. 상태 및 동작 시퀀스
1. **MoveToPose**  
   프리셋 포즈 복사 → 각 팔에 `_send_pose` 호출 → `in_progress` 상태 발행 → 완료 후 `complete`.
2. **PickProduct**  
   접근 포즈(`z + approach_offset`) → 목표 포즈 → 그리퍼 닫기 → 상승(`z + lift_offset`) → `completed`.
3. **PlaceProduct**  
   접근 포즈 → 목표 포즈 → 그리퍼 열기 → 상승 → `completed`.
4. 상태 토픽의 `status`/`current_phase` 값은 Interface Specification 문서와 동일하다.

## 7. 예외 및 로깅
- 시리얼 연결 실패, 좌표 전송 실패, 그리퍼 오류 발생 시 에러 로그를 남기고 `success=False`를 반환한다.
- 파라미터 형식 오류는 경고 로그 후 기본값을 사용한다.
- 서비스 수행 중 예외가 발생하면 상태 토픽 `failed` 이벤트를 발행한다.

## 8. 파라미터 요약

| 구분 | 노드 | 파라미터 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| 하드웨어 | 듀얼 | `enabled_arms` | `left,right` | 활성화할 팔 목록 |
| 하드웨어 | 듀얼 | `serial_port_left` / `serial_port_right` | `/dev/ttyUSB0` / `/dev/ttyUSB1` | 각 팔 시리얼 포트 |
| 통신 | 듀얼/단일 | `baud_rate` | `1000000` | 직렬 통신 속도 |
| 동작 | 듀얼/단일 | `move_speed` | `40` | `sync_send_coords` 속도 (0~100) |
| 동작 | 듀얼/단일 | `approach_offset_m` | `0.05` | 접근 시 추가 높이 (m) |
| 동작 | 듀얼/단일 | `lift_offset_m` | `0.06` | 픽업/담기 후 상승 높이 (m) |
| 프리셋 | 듀얼/단일 | `preset_pose_cart_view` | `[0.16, 0.0, 0.18, 0.0, 0.0, 0.0]` | 카트 확인 자세 |
| 프리셋 | 듀얼/단일 | `preset_pose_standby` | `[0.10, 0.0, 0.14, 0.0, 0.0, 0.0]` | 기본 대기 자세 |
| 그리퍼 | 듀얼/단일 | `gripper_open_value` / `gripper_close_value` | `100` / `20` | 그리퍼 개방/파지 값 |
| 토픽 | 듀얼/단일 | `pose_status_topic` | `/packee/arm/pose_status` | 자세 상태 토픽 |
| 토픽 | 듀얼/단일 | `pick_status_topic` | `/packee/arm/pick_status` | 픽업 상태 토픽 |
| 토픽 | 듀얼/단일 | `place_status_topic` | `/packee/arm/place_status` | 담기 상태 토픽 |
| 브릿지 | 공통 | `left_velocity_topic` / `right_velocity_topic` | `/packee/jetcobot/left/cmd_vel` / `/packee/jetcobot/right/cmd_vel` | 속도 명령 토픽 |
| 브릿지 | 공통 | `command_period_sec` | `0.15` | 속도 적분 주기 |
| 브릿지 | 공통 | `workspace_radial` | `0.28` | 허용 수평 반경 |
| 브릿지 | 공통 | `workspace_z_min` / `workspace_z_max` | `0.05` / `0.30` | 허용 Z 범위 |

## 9. 테스트 전략
- **서비스 수동 호출** : `ros2 service call /packee/arm/pick_product ...` 형태로 좌표·진행률을 검증한다.
- **토픽 모니터링** : `ros2 topic echo /packee/arm/pick_status`로 단계별 메시지를 확인한다.
- **브릿지 검증** : Twist/Float32 명령을 수동 발행해 좌표 적분과 보정 동작을 확인한다.
- **HIL 테스트** : 실제 myCobot 280 듀얼 암을 사용해 픽업/담기 반복 및 안전 오프셋을 검증한다.

## 10. 향후 보완 사항
- 좌/우 팔 간 충돌 방지를 위한 간섭 검사 로직 추가.
- 상태 토픽에 에러 코드 및 진단 정보 확장.
- Visual Servo 및 경로 계획 기능을 Python 모듈로 단계적 도입.
- 서비스 처리 중 비동기 작업 관리(스레드/asyncio) 도입을 통한 응답성 향상.
