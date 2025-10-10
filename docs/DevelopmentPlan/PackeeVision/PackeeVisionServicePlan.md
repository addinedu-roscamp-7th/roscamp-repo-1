# Packee Vision AI Service 개발 계획

## 1단계: 기반 설정 (Foundation)
- **목표**: Vision 서비스 제공을 위한 ROS2 노드 및 카메라 연동
- **세부 작업**:
  1. **ROS2 노드 생성**: `packee_vision_service` 노드 생성
  2. **카메라 연동**: Packee 로봇에 부착된 카메라의 ROS2 드라이버를 연동하고, 이미지 토픽을 정상적으로 수신하는지 확인
  3. **AI 모델 라이브러리 설치**: OpenCV, PyTorch/TensorFlow 등 AI 모델 추론에 필요한 라이브러리 설치

## 2단계: AI 모델 통합 (Model Integration)
- **목표**: 학습된 AI 모델을 로드하여 이미지 추론 기능 구현
- **세부 작업**:
  1. **객체 탐지 모델**: 장바구니 안에 있는 다양한 상품들을 탐지하는 딥러닝 모델(e.g., YOLO) 로드
  2. **상태 인식 모델**: 장바구니의 존재 유무, 장바구니가 비었는지 여부 등을 판단하는 모델 또는 규칙 기반 로직 구현

## 3단계: 인터페이스 구현 (Interface)
- **목표**: `Pac_Main_vs_Pac_Vision.md` 명세에 따른 인터페이스 구현
- **세부 작업**:
  1. **서비스 서버 구현**: `/check_cart_presence`, `/detect_products_in_cart`, `/verify_packing_complete` 등 Main Controller로부터 오는 서비스 요청을 처리하고, AI 추론 후 결과를 응답하는 서버 구현
  2. **3D 위치 추정**: 2D 이미지의 Bounding Box와 Depth 카메라 정보를 결합하여, 장바구니 안 상품들의 3차원 공간 좌표를 계산하는 로직 구현. 이 좌표는 Arm Controller가 상품을 집는 데 사용됨.
