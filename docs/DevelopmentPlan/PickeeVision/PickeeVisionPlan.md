# Pickee Vision AI Service 개발 계획

---

## 1. 개요

본 문서는 Pickee 로봇의 시각 인지 기능을 담당하는 Pickee Vision AI Service의 개발 계획을 정리합니다. 주요 역할은 주행, 상품 관리, 직원 상호작용 등 다양한 시나리오에 필요한 시각 정보를 분석하여 Pickee Main Controller에 제공하는 것입니다.

## 2. 주요 기능

- 장애물 감지 및 정보 발행
- 매대 상품 및 장바구니 내 상품 인식
- 장바구니 존재 유무 확인
- 직원 등록, 추종 및 위치 정보 제공
- 원격 모니터링을 위한 실시간 영상 스트리밍

## 3. 아키텍처 설계: 노드 분리

유지보수성과 확장성을 고려하여, Pickee Vision의 기능은 다음과 같이 4개의 독립적인 ROS2 노드로 분리하여 설계합니다.

- **`obstacle_detector_node`**: 주행 중 장애물 감지 및 정보 발행 전담
- **`product_detector_node`**: 상품 및 장바구니 관련 인식 서비스 및 결과 발행 전담
- **`staff_tracker_node`**: 직원 등록, 추종 등 상호작용 시나리오 전담
- **`camera_service_node`**: 영상 스트리밍 제어 및 하드웨어 관련 서비스 전담

## 4. 세부 구현 계획: 노드별 인터페이스

각 노드는 다음과 같은 ROS2 인터페이스를 구현합니다.

#### 1) `obstacle_detector_node`
- **토픽 발행 (Publisher)**
    - `/pickee/vision/obstacle_detected`

#### 2) `product_detector_node`
- **서비스 제공 (Service Server)**
    - `/pickee/vision/detect_products`
    - `/pickee/vision/check_product_in_cart`
    - `/pickee/vision/check_cart_presence`
- **토픽 발행 (Publisher)**
    - `/pickee/vision/detection_result`
    - `/pickee/vision/cart_check_result`

#### 3) `staff_tracker_node`
- **서비스 제공 (Service Server)**
    - `/pickee/vision/set_mode`
    - `/pickee/vision/register_staff`
    - `/pickee/vision/track_staff`
- **토픽 발행 (Publisher)**
    - `/pickee/vision/register_staff_result`
    - `/pickee/vision/staff_location`
- **서비스 호출 (Service Client)**
    - `/pickee/tts_request`

#### 4) `camera_service_node`
- **서비스 제공 (Service Server)**
    - `/pickee/vision/video_stream_start`
    - `/pickee/vision/video_stream_stop`
- **내부 기능**
    - UDP 영상 스트림 송출 (포트 `6000`)

## 5. 개발 단계 (초안)

#### 1단계: 스켈레톤 구현 및 통신 테스트 (현재 완료)
- ROS2 패키지 및 4개 노드 기본 구조 생성
- `shopee_interfaces` 기반 모든 서비스/토픽 인터페이스 스켈레톤 구현
- Mock 데이터를 사용한 인터페이스 통신 검증

#### 2단계: 핵심 로직 구현
- 각 노드에 실제 AI 모델(YOLO 등) 연동
- 카메라 및 하드웨어 드라이버 연동
- UDP 영상 스트리밍 클라이언트 상세 구현 및 `camera_service_node`와 통합

#### 3단계: 통합 테스트 및 안정화
- `Pickee Main`과의 연동 테스트 수행
- 실제 로봇 환경에서의 필드 테스트 및 성능 튜닝

## 6. 의존성

- `rclpy`: ROS2 Python 클라이언트 라이브러리
- `shopee_interfaces`: 프로젝트 공용 메시지 및 서비스 패키지
- `(추가 예정)` AI 모델 라이브러리 (e.g., PyTorch, TensorFlow)
- `(추가 예정)` 카메라 드라이버 라이브러리 (e.g., OpenCV)


