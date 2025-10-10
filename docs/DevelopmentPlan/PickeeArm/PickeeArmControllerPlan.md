# Pickee Arm Controller 개발 계획

## 1단계: 기반 설정 (Foundation)
- **목표**: 로봇 팔 제어를 위한 ROS2 노드 및 하드웨어 연동
- **세부 작업**:
  1. **ROS2 노드 생성**: `pickee_arm_controller` 노드 생성
  2. **하드웨어 드라이버 연동**: 로봇 팔(Manipulator) 및 그리퍼의 ROS2 드라이버를 연동하고 제어 테스트

## 2단계: 모션 플래닝 적용 (Motion Planning)
- **목표**: MoveIt 2를 이용한 충돌 회피 및 경로 계획 기능 구현
- **세부 작업**:
  1. **MoveIt 설정**: 로봇의 URDF(Unified Robot Description Format) 파일을 기반으로 MoveIt 설정 파일 생성
  2. **모션 플래닝 테스트**: 특정 좌표 또는 지정된 자세(Pose)로 팔을 움직이는 경로를 생성하고 실행하는 기능 테스트

## 3단계: 인터페이스 구현 (Interface)
- **목표**: `Pic_Main_vs_Pic_Arm.md` 명세에 따른 인터페이스 구현
- **세부 작업**:
  1. **서비스 서버 구현**: `/pickee/arm/move_to_pose`, `/pick_product`, `/place_product` 서비스 요청을 받아 MoveIt으로 팔 동작을 수행하는 서버 구현
  2. **토픽 발행 구현**: 팔의 자세 변경, 픽업, 담기 등 각 동작의 진행 상태(`in_progress`, `completed`, `failed`)를 `/pickee/arm/pose_status`, `/pick_status`, `/place_status` 토픽으로 발행(Publish)

## 4단계: 정밀 제어 고도화 (Fine-tuning)
- **목표**: Vision 데이터와 연동한 정밀한 제어 구현
- **세부 작업**:
  1. **Grasp Planning**: Vision이 제공하는 상품의 3D 위치 및 BBox 정보를 바탕으로 안정적인 파지(Grasping) 계획을 생성하는 로직 구현
  2. **실시간 보정**: 상품을 집거나 놓을 때 발생할 수 있는 오차를 보정하기 위한 제어 로직 고도화
