# Shopee ROS2 Workspace

**Shopee 로봇 쇼핑 시스템**의 ROS2 워크스페이스입니다. 자율 주행 로봇을 활용한 원격 쇼핑 및 상품 포장 서비스를 제공하는 통합 플랫폼입니다.

## 🏗️ 시스템 아키텍처

본 시스템은 **Pickee**(쇼핑 로봇)와 **Packee**(포장 로봇) 두 종류의 로봇이 협업하여 고객의 원격 쇼핑 요청을 처리합니다.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Shopee App    │◄──►│ Main Service     │◄──►│   LLM Service   │
│   (Qt/ROS2)     │    │ (중앙 제어)      │    │ (자연어 처리)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │    Pickee     │       │    Packee     │
            │  (쇼핑 로봇)  │       │  (포장 로봇)  │
            │               │       │               │
            │ • 상품 피킹   │       │ • 상품 포장   │
            │ • 자율 주행   │       │ • 듀얼 암     │
            │ • 실시간 영상 │       │ • 품질 검증   │
            └───────────────┘       └───────────────┘
```

## 📦 패키지 구성

### 🤖 로봇 제어 패키지

#### Pickee (쇼핑 로봇)
- **`pickee_main`** - Pickee 메인 컨트롤러 및 상태 머신
- **`pickee_mobile`** - 자율 주행 및 경로 계획
- **`pickee_vision`** - 상품/장애물/직원 인식
- **`pickee_arm`** - 로봇팔 제어 (개발 예정)

#### Packee (포장 로봇)  
- **`packee_main`** - Packee 메인 컨트롤러 및 서비스 노드
- **`packee_vision`** - 장바구니 및 상품 인식
- **`packee_arm`** - 듀얼 암 협업 제어

### 🌐 서비스 패키지
- **`main_service`** - 중앙 백엔드 서비스 (TCP/UDP/ROS2 통신)
- **`shopee_interfaces`** - ROS2 메시지/서비스 인터페이스 정의
- **`shopee_app`** - Qt 기반 사용자 인터페이스

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
# ROS2 Jazzy 설치 (미설치시)
sudo apt update && sudo apt install ros-jazzy-desktop

# 추가 ROS2 패키지
sudo apt install ros-jazzy-cv-bridge ros-jazzy-image-transport \
                 ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
                 ros-jazzy-moveit ros-jazzy-joint-state-publisher

# Python 의존성
pip3 install opencv-python numpy scipy matplotlib \
             mysql-connector-python pymysql sqlalchemy \
             fastapi uvicorn requests asyncio \
             pyqt6 watchdog
```

### 2. 워크스페이스 빌드

```bash
# 워크스페이스로 이동
cd ~/dev_ws/Shopee/shopee_ros2

# 의존성 설치
rosdep install --from-paths src --ignore-src -r -y

# 전체 빌드
colcon build

# 환경 설정
source install/setup.bash
```

### 3. 시스템 실행

#### 기본 서비스 실행
```bash
# 터미널 1: 메인 서비스
ros2 run main_service main_service_node

# 터미널 2: Pickee 메인 컨트롤러  
ros2 run pickee_main main_controller

# 터미널 3: Packee 비전 서비스
ros2 run packee_vision cart_presence_checker
```

#### GUI 애플리케이션 실행
```bash
cd src/shopee_app
python3 app.py
```

## 🎯 주요 기능

### 원격 쇼핑 워크플로우
1. **고객 로그인** - 사용자 인증 및 프로필 로드
2. **상품 검색** - 자연어/음성 기반 상품 검색
3. **실시간 선택** - 로봇 카메라를 통한 신선식품 선택
4. **자동 피킹** - Pickee 로봇의 자율 상품 수집
5. **스마트 포장** - Packee 로봇의 듀얼암 협업 포장

### 로봇 자율 기능
- **SLAM 기반 자율 주행** - 실시간 맵핑 및 경로 계획
- **동적 장애물 회피** - 사람, 카트 등 실시간 회피
- **컴퓨터 비전** - YOLOv8 기반 상품/객체 인식
- **자동 충전** - 배터리 모니터링 및 자동 도킹

## 🔧 개발 가이드

### 코딩 표준
본 프로젝트는 엄격한 코딩 표준을 준수합니다. 자세한 내용은 [`docs/CodingStandard/standard.md`](../docs/CodingStandard/standard.md)를 참조하세요.

#### 주요 규칙
- **ROS2**: Package/Node/Topic/Service 이름은 `snake_case`
- **Python**: 함수/변수는 `snake_case`, 클래스는 `PascalCase`  
- **C++**: 함수는 `PascalCase`, 변수는 `snake_case`
- **주석**: 한국어 사용, C++은 `//`, Python은 `#`

### 테스트 실행
```bash
# 단위 테스트
cd src/main_service && pytest
cd src/pickee_main && python3 -m pytest

# 통합 테스트  
cd src/pickee_main && ./run_tests.sh

# ROS2 테스트
colcon test --packages-select pickee_main
colcon test-result --verbose
```

### 패키지별 개발 가이드
- **Main Service**: [`src/main_service/README.md`](src/main_service/README.md)
- **Pickee Main**: [`src/pickee_main/README.md`](src/pickee_main/README.md)  
- **Interfaces**: [`src/shopee_interfaces/README.md`](src/shopee_interfaces/README.md)

## 📊 시스템 요구사항

본 시스템은 **원격 쇼핑**, **자율 주행**, **상품 포장** 등 25개의 시스템 요구사항을 구현합니다.

### 주요 기능 영역
- **계정 및 상품 관리**: 로그인, 상품 검색/예약, 결제
- **원격 쇼핑**: 실시간 상품 선택, 영상 모니터링, 알림
- **로봇 제어**: 자율 주행, 장애물 회피, 상품 피킹/포장
- **모니터링**: 주문/작업/로봇 상태 실시간 조회 및 이력 관리
- **자율 기능**: 자동 복귀, 자동 충전 (일부 개발 중)

상세한 요구사항 명세는 [시스템 요구사항 문서](../docs/Requirements/SystemRequirements.md)를 참조하세요.

## 🗂️ 디렉토리 구조

```
shopee_ros2/
├── src/                          # ROS2 소스 패키지
│   ├── pickee_main/              # Pickee 메인 컨트롤러
│   ├── pickee_mobile/            # Pickee 모바일 베이스
│   ├── pickee_vision/            # Pickee 비전 시스템
│   ├── packee_main/              # Packee 메인 컨트롤러
│   ├── packee_vision/            # Packee 비전 시스템  
│   ├── packee_arm/               # Packee 듀얼 암
│   ├── main_service/             # 중앙 백엔드 서비스
│   ├── shopee_interfaces/        # ROS2 인터페이스
│   └── shopee_app/               # Qt GUI 애플리케이션
├── build/                        # 빌드 아티팩트
├── install/                      # 설치된 패키지
├── log/                          # 빌드 로그
└── README.md                     # 이 파일
```

## 🔗 관련 문서

### 설계 문서
- [시스템 요구사항](../docs/Requirements/SystemRequirements.md)
- [소프트웨어 아키텍처](../docs/Architecture/SWArchitecture.md)
- [하드웨어 아키텍처](../docs/Architecture/HWArchitecture.md)
- [데이터베이스 설계](../docs/ERDiagram/ERDiagram.md)

### 개발 계획
- [메인 서비스](../docs/DevelopmentPlan/MainService/)
- [Pickee 로봇](../docs/DevelopmentPlan/PickeeMain/)
- [Packee 로봇](../docs/DevelopmentPlan/PackeeMain/)

### 인터페이스 명세
- [App ↔ Main Service](../docs/InterfaceSpecification/App_vs_Main.md)
- [Main ↔ Pickee](../docs/InterfaceSpecification/Main_vs_Pic_Main.md)
- [Main ↔ Packee](../docs/InterfaceSpecification/Main_vs_Pac_Main.md)