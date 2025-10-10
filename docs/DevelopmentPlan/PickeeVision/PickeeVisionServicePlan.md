# Pickee Vision AI Service 개발 계획

## 1단계: 기반 설정 (Foundation)
- **목표**: Vision 서비스 제공을 위한 ROS2 노드 및 카메라 연동
- **세부 작업**:
  1. **ROS2 노드 생성**: `pickee_vision_service` 노드 생성
  2. **카메라 연동**: 로봇에 부착된 카메라(전방, 팔)의 ROS2 드라이버를 연동하고, 이미지 토픽을 정상적으로 수신하는지 확인
  3. **AI 모델 라이브러리 설치**: OpenCV, PyTorch/TensorFlow 등 AI 모델 추론에 필요한 라이브러리 설치

## 2단계: AI 모델 통합 (Model Integration)
- **목표**: 학습된 AI 모델을 로드하여 이미지 추론 기능 구현
- **세부 작업**:
  1. **객체 탐지 모델**: 상품(신선식품, 공산품), 장애물(카트, 사람) 등을 탐지하는 YOLO, SSD 등 딥러닝 모델 로드
  2. **특징 추출 모델**: 직원 등록 및 추종을 위한 외형 특징(feature) 추출 모델(e.g., ResNet, ViT) 로드

## 3단계: 인터페이스 구현 (Interface)
- **목표**: `Pic_Main_vs_Pic_Vision.md` 명세에 따른 인터페이스 구현
- **세부 작업**:
  1. **서비스 서버 구현**: `/detect_products`, `/check_product_in_cart` 등 Main Controller로부터 오는 서비스 요청을 처리하고, AI 모델 추론을 수행한 뒤 결과를 응답하는 서버 구현
  2. **토픽 발행 구현**: 추론 결과를 `/detection_result`, `/cart_check_result` 등 관련 토픽으로 발행(Publish)
  3. **실시간 처리**: 장애물, 추종 직원 위치 등 실시간으로 처리해야 하는 데이터를 지속적으로 발행하는 로직 구현

## 4단계: Vision 로직 고도화 (Enhancement)
- **목표**: 3D 위치 추정 등 Vision 서비스의 정확도 및 기능 향상
- **세부 작업**:
  1. **3D 위치 추정**: 2D 이미지의 Bounding Box와 Depth 카메라 정보를 결합하여, 객체의 3차원 공간 좌표를 계산하는 로직 구현
  2. **직원 등록/추종**: 직원의 정면/후면 특징을 등록하고, 이를 바탕으로 특정 직원을 식별하고 추적하는 전체 시나리오 구현
