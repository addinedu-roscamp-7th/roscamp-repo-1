# Pickee Mobile Controller 개발 계획

## 1단계: 기반 설정 (Foundation)
- **목표**: 모바일 베이스 제어를 위한 ROS2 노드 및 하드웨어 연동
- **세부 작업**:
  1. **ROS2 노드 생성**: `pickee_mobile_controller` 노드 생성
  2. **하드웨어 드라이버 연동**: 모터 컨트롤러, LiDAR 등 센서의 ROS2 드라이버를 연동하고 데이터 수신 확인

## 2단계: 내비게이션 스택 적용 (Navigation)
- **목표**: ROS2 Navigation Stack(Nav2)을 이용한 자율 주행 기능 구현
- **세부 작업**:
  1. **SLAM**: SLAM(동시적 위치 추정 및 지도 작성)을 수행하여 마트 환경 지도 생성
  2. **Nav2 설정**: 생성된 지도를 바탕으로 Nav2의 Costmap, Controller, Planner 등 파라미터 설정
  3. **경로 추종 테스트**: Nav2를 통해 특정 목적지까지 경로를 생성하고 주행하는 기능 테스트

## 3단계: 인터페이스 구현 (Interface)
- **목표**: `Pic_Main_vs_Pic_Mobile.md` 명세에 따른 인터페이스 구현
- **세부 작업**:
  1. **서비스 서버 구현**: `/pickee/mobile/move_to_location`, `/update_global_path` 서비스 요청을 받아 Nav2에 목표를 전달하는 서버 구현
  2. **토픽 발행/구독 구현**:
     - 로봇의 현재 위치, 속도, 배터리 상태 등을 `/pickee/mobile/pose` 토픽으로 주기적으로 발행(Publish)
     - `/pickee/mobile/speed_control` 토픽을 구독(Subscribe)하여 주행 속도를 동적으로 제어하는 로직 구현
     - 목적지 도착 시 `/pickee/mobile/arrival` 토픽 발행

.
