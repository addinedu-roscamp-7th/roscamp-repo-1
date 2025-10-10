# Pickee Main Controller 개발 계획

## 1단계: 노드 및 상태 기계 설정 (Node & State Machine)
- **목표**: Pickee 로봇의 두뇌 역할을 할 메인 컨트롤러 노드 설정
- **세부 작업**:
  1. **ROS2 노드 생성**: `rclpy`를 사용하여 `pickee_main_controller` 노드 생성
  2. **상태 기계 구현**: `StateDefinition.md`에 정의된 Pickee의 상태(MOVING_TO_SHELF, PICKING_PRODUCT 등)를 관리하는 상태 기계(State Machine) 구현

## 2단계: 외부 시스템 연동 (External Integration)
- **목표**: Main Service 및 LLM Service와의 통신 구현
- **세부 작업**:
  1. **Main Service 연동**: `Main_vs_Pic_Main.md` 명세에 따라, Main Service가 보내는 서비스 요청(e.g., `/start_task`)을 처리하는 서버(Server) 구현 및 상태 보고 토픽 발행(Publish)
  2. **LLM Service 연동**: (필요시) 직원 음성 명령 분석을 위해 LLM 서비스 API를 호출하는 HTTP 클라이언트 구현

## 3단계: 내부 컴포넌트 연동 (Internal Orchestration)
- **목표**: Pickee의 Mobile, Arm, Vision 컴포넌트를 총괄 제어
- **세부 작업**:
  1. **Mobile 연동**: `/pickee/mobile/move_to_location` 서비스 클라이언트 구현 및 `/arrival` 토픽 구독
  2. **Arm 연동**: `/pickee/arm/move_to_pose`, `/pick_product` 등 서비스 클라이언트 구현 및 상태 토픽 구독
  3. **Vision 연동**: `/pickee/vision/detect_products` 등 서비스 클라이언트 구현 및 결과 토픽 구독

## 4단계: 시나리오별 워크플로우 구현 (Workflow Implementation)
- **목표**: 실제 시나리오에 맞춰 전체 작업 흐름 구현
- **세부 작업**:
  1. **원격 쇼핑 워크플로우**: `start_task` 명령 수신 -> Mobile 제어 (이동) -> Arm/Vision 제어 (인식/피킹) -> Main Service 보고
  2. **재고 보충 워크플로우**: 직원 추종, 음성 명령 인식, 창고-매대 이동 등 관련 로직 구현
