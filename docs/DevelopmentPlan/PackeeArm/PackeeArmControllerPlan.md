# Packee Arm Controller 개발 계획

## 1단계: 기반 설정 (Foundation)
- **목표**: 양팔 로봇 제어를 위한 ROS2 노드 및 하드웨어 연동
- **세부 작업**:
  1. **ROS2 노드 생성**: `packee_arm_controller` 노드 생성
  2. **하드웨어 드라이버 연동**: 양팔(Dual-Arm) 및 그리퍼의 ROS2 드라이버를 연동하고 개별 제어 테스트

## 2단계: 양팔 모션 플래닝 (Dual-Arm Motion Planning)
- **목표**: MoveIt 2를 이용한 양팔 협응 및 충돌 회피 기능 구현
- **세부 작업**:
  1. **MoveIt 설정**: 양팔 로봇의 URDF 파일을 기반으로 MoveIt 설정 파일 생성. 특히 양팔 간, 그리고 로봇 자신과의 충돌을 방지하는 설정이 중요.
  2. **협응 테스트**: 한 팔이 다른 팔의 작업 공간을 침범하지 않으면서, 동시에 작업을 수행하거나 순차적으로 작업을 수행하는 경로 계획 기능 테스트

## 3단계: 인터페이스 구현 (Interface)
- **목표**: `Pac_Main_vs_Pac_Arm.md` 명세에 따른 인터페이스 구현
- **세부 작업**:
  1. **서비스 서버 구현**: `/packee/arm/move_to_pose`, `/pick_product`, `/place_product` 서비스 요청을 받아, 지정된 팔(`arm_side`)이 동작을 수행하도록 MoveIt API를 호출하는 서버 구현
  2. **토픽 발행 구현**: 각 팔의 자세 변경, 픽업, 담기 등 동작의 진행 상태를 `/packee/arm/pose_status`, `/pick_status`, `/place_status` 토픽으로 발행

## 4단계: 포장 작업 고도화 (Fine-tuning)
- **목표**: 다양한 형태의 상품을 안정적으로 집고 놓는 기능 고도화
- **세부 작업**:
  1. **Grasp Planning**: Vision이 제공하는 상품 정보를 바탕으로, 상품의 형태와 무게 중심을 고려하여 최적의 파지(Grasping) 자세와 힘을 결정하는 로직 구현
  2. **Place Planning**: 포장 박스 내에 상품을 순서대로, 그리고 안정적으로 쌓기 위한 최적의 위치와 자세를 결정하는 로직 구현
