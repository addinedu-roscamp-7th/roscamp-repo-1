# Shopee Packee Arm Controller 구현 계획

본 계획서는 Python 기반 Packee Arm 패키지(`pymycobot_dual`, `pymycobot_left`, `pymycobot_right`, `jetcobot_bridge.py`)를 구현·테스트하기 위한 단계별 절차를 정의한다. Packee Main Controller와 합의된 인터페이스와 코딩 표준(`docs/CodingStandard/standard.md`)을 준수하며, 모든 산출물은 `docs/DevelopmentPlan/PackeeArm/`에 기록한다.

## Phase 1: 환경 정비 및 기본 골격
**목표**: Python 패키지 구조와 런치 환경을 준비하고 필수 파라미터를 정리한다.
- **Step 1.1 패키지 구조 점검**
  - `package.xml`을 rclpy 중심 의존성으로 정리하고, `CMakeLists.txt`에서 Python 실행 파일 설치 경로를 명확히 한다.
  - `resource/packee_arm`의 `package_type`을 `ament_cmake`로 설정해 `ros2 run` 호환성을 확보한다.
- **Step 1.2 코딩 표준 정비**
  - Python 스크립트의 문자열, 들여쓰기, 주석 규칙을 표준과 비교 확인하고, 자동화 도구(pyproject, lint)는 미도입 시 수동 점검 프로세스 기록.
- **Step 1.3 런치 인자 정의**
  - 듀얼/단일 팔 모드 플래그(`run_pymycobot_dual`, `run_pymycobot_left/right`), `pymycobot_enabled_arms`(기본 `'left,right'`), 프리셋 자세, 시리얼 포트 등을 문서화하고 기본값을 확정한다.

## Phase 2: 좌/우 팔 서비스 노드 구현
**목표**: Packee Main이 호출하는 서비스와 상태 토픽을 Python 노드에서 처리한다.
- **Step 2.1 서비스 파이프라인**
  - `pymycobot_dual`에서 `/packee/arm/move_to_pose`, `/packee/arm/pick_product`, `/packee/arm/place_product` 콜백을 구현한다.
  - Pose6D 파라미터를 `[x, y, z, rx, ry, rz]` 형식으로 변환하고, 접근/상승 오프셋을 적용한다.
  - arm_side 기반으로 좌/우 팔을 분기하고, MoveToPose는 활성화된 모든 팔에 적용되도록 구현한다.
- **Step 2.2 단일 팔 옵션**
  - 필요 시 `pymycobot_left`, `pymycobot_right`를 재사용해 단일 팔 전용 노드를 제공한다.
  - 동일한 서비스 이름을 사용하므로 듀얼 노드와 동시에 기동되지 않도록 런치 제약을 문서화한다.
- **Step 2.3 상태 토픽 발행**
  - 서비스 단계별 진행률과 메시지를 `/packee/arm/pose_status`, `/packee/arm/pick_status`, `/packee/arm/place_status`로 발행한다.
  - `current_phase`와 `status` 값은 Interface Specification 문서와 동일하게 유지한다.
- **Step 2.4 예외 처리**
  - 하드웨어 미연결, 파라미터 형식 오류, pymycobot 예외 발생 시 로그 기록과 `success=False` 응답을 정의한다.

## Phase 3: JetCobot 브릿지 구현
**목표**: 상위 모듈이 발행하는 Twist/Float32 명령을 myCobot 시리얼 명령으로 변환한다.
- **Step 3.1 파라미터 정의**
  - `command_period_sec`, `workspace_*`, `default_pose_*`, `move_speed`, 그리퍼 파라미터를 선언한다.
  - 문자열로 전달된 프리셋도 허용하도록 파라미터 파싱 유틸리티 작성.
- **Step 3.2 속도 → 좌표 변환**
  - Twist 속도를 적분해 `(x, y, z, rx, ry, rz)` 좌표를 누적하고, 라디안을 degree로 변환하여 `sync_send_coords` 호출.
  - 작업 공간을 벗어나는 경우 자동 보정(스케일링/클램프) 로직 적용.
- **Step 3.3 그리퍼 명령 처리**
  - Float32 명령을 수신해 `set_gripper_value` 호출, 에러 발생 시 로그 처리.

## Phase 4: 통합 테스트 및 문서화
**목표**: 서비스와 브릿지를 통합 검증하고 운영 가이드를 정리한다.
- **Step 4.1 서비스/토픽 수동 검증**
  - `ros2 service call` / `ros2 topic echo`를 사용해 표준 시나리오(자세 이동, 픽업, 담기)를 재현한다.
- **Step 4.2 브릿지 기능 점검**
  - `ros2 topic pub`을 이용해 Twist/Float32 명령을 주입하고, myCobot 좌표 변환 및 작업 공간 보정이 정상동작하는지 확인한다.
- **Step 4.3 문서 업데이트**
  - 테스트 절차를 `TEST_GUIDE.md`에 반영하고, 실제 하드웨어 사용 없이도 검증 가능한 Stub/Mock 절차를 명시한다.

## Phase 5: HIL 및 운영 준비
**목표**: 실제 myCobot 280 듀얼 암을 대상으로 안정성을 확인하고 운영 체크리스트를 작성한다.
- **Step 5.1 HIL 테스트**
  - 좌/우 팔 각각 10회 이상 픽업/담기 반복, 안전 오프셋과 작업 공간 한계 검증.
- **Step 5.2 장애 대응 가이드**
  - 시리얼 연결 실패, 좌표 전송 오류, 그리퍼 오류에 대한 대응 절차와 로그 패턴을 문서화한다.
- **Step 5.3 인수 기준(DoD)**
  - 서비스 응답과 상태 토픽이 명세대로 발행될 것
  - 런치 인자 조합(좌/우/브릿지) 별 기동 확인
  - 코딩 표준 및 문서 최신화 완료

## 공통 관리 항목
- **의존성**: `pymycobot` Python 라이브러리 설치, `/packee/arm/*` 서비스 및 토픽 메시지 스키마 유지.
- **리스크 및 대응**
  - 하드웨어 미연결 → 모의 테스트 절차 마련, 로그 레벨 구분
  - 좌/우 팔 충돌 및 서비스 충돌 → 작업 공간 제한 값 조정, 런치 인자 가이드로 단일 팔만 기동하도록 안내
  - 시리얼 지연 → `move_speed`, `command_period_sec` 튜닝 및 로그 모니터링
- **커뮤니케이션**: Packee Main/Pickee 팀과 주간 동기화, 하드웨어 팀과 포트/배선 변경 시 공유, QA 테스트 결과를 공용 문서로 갱신한다.
