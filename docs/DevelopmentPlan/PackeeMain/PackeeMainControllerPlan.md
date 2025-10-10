# Packee Main Controller 개발 계획

## 1단계: 노드 및 상태 기계 설정 (Node & State Machine)
- **목표**: Packee 로봇의 두뇌 역할을 할 메인 컨트롤러 노드 설정
- **세부 작업**:
  1. **ROS2 노드 생성**: `packee_main_controller` 노드 생성
  2. **상태 기계 구현**: `StateDefinition.md`에 정의된 Packee의 상태(STANDBY, CHECKING_CART, PACKING_PRODUCTS 등)를 관리하는 상태 기계(State Machine) 구현

## 2단계: 외부/내부 연동 (Integration)
- **목표**: Main Service 및 Packee의 Arm, Vision 컴포넌트와의 통신 구현
- **세부 작업**:
  1. **Main Service 연동**: `Main_vs_Pac_Main.md` 명세에 따라, Main Service의 서비스 요청을 처리하고 포장 상태 및 결과를 토픽으로 발행
  2. **Vision 연동**: `/packee/vision/check_cart_presence`, `/detect_products_in_cart` 등 서비스 클라이언트 구현
  3. **Arm 연동**: `/packee/arm/move_to_pose`, `/pick_product`, `/place_product` 등 서비스 클라이언트 구현 및 상태 토픽 구독

## 3단계: 포장 워크플로우 구현 (Packing Workflow)
- **목표**: 실제 포장 시나리오에 맞춰 전체 작업 흐름 구현
- **세부 작업**:
  1. **작업 계획 로직**: Vision으로부터 받은 장바구니 내 상품 목록을 바탕으로, 양팔의 작업 순서(Task Scheduling) 및 충돌 없는 경로(Collision-Free Plan)를 계획하는 로직 구현
  2. **포장 실행 로직**: 계획된 작업 순서에 따라 Arm과 Vision에 순차적으로 명령을 내리고, 모든 상품을 포장 박스로 옮기는 전체 워크플로우 구현
  3. **완료 검증**: 모든 상품을 옮긴 후, Vision 서비스를 호출하여 장바구니가 비었는지 최종 확인하는 로직 구현
