# Shopee 프로젝트 설계 문서

[![Documentation](https://img.shields.io/badge/Documentation-Complete-green.svg)](.)
[![Architecture](https://img.shields.io/badge/Architecture-Designed-blue.svg)](Architecture/)
[![Requirements](https://img.shields.io/badge/Requirements-Defined-orange.svg)](Requirements/)
[![Interfaces](https://img.shields.io/badge/Interfaces-Specified-purple.svg)](InterfaceSpecification/)

**Shopee 로봇 쇼핑 시스템**의 종합 설계 문서 저장소입니다. 시스템 요구사항부터 상세 구현 계획까지 프로젝트의 모든 설계 정보를 체계적으로 관리합니다.

## 📋 문서 개요

본 문서 저장소는 **자율 주행 로봇을 활용한 원격 쇼핑 서비스**의 전체 시스템 설계를 다룹니다. Pickee(쇼핑 로봇)와 Packee(포장 로봇)가 협업하여 고객에게 완전한 원격 쇼핑 경험을 제공하는 통합 플랫폼의 설계 명세서입니다.

### 🎯 프로젝트 목표
- **원격 쇼핑**: 고객이 집에서 실시간으로 상품을 선택하고 구매
- **자율 로봇**: AI 기반 자율 주행 및 상품 인식 로봇 시스템
- **스마트 포장**: 듀얼 암 협업을 통한 지능형 상품 포장
- **통합 관리**: 실시간 모니터링 및 통합 관제 시스템

## 🗂️ 문서 구조

### 📊 [Requirements](Requirements/) - 요구사항 정의
시스템의 기능적/비기능적 요구사항을 정의합니다.

- **[SystemRequirements.md](Requirements/SystemRequirements.md)** - 시스템 요구사항 (SR_01~SR_25)
  - 계정 관리, 상품 탐색, 원격 쇼핑
  - 실시간 모니터링, 상품 포장, 재고 보충
  - 로봇 자율 기능 (주행, 충전, 복귀)

- **[UserRequirements.md](Requirements/UserRequirements.md)** - 사용자 요구사항 (UR_01~UR_13)
  - 고객: 계정 관리, 상품 탐색, 원격 쇼핑, 모니터링
  - 직원: 상품 포장 보조, 재고 보충 보조
  - 관리자: 주문/작업/로봇/상품 정보 관리

### 🏗️ [Architecture](Architecture/) - 시스템 아키텍처
전체 시스템의 하드웨어 및 소프트웨어 구조를 설계합니다.

- **[HWArchitecture.md](Architecture/HWArchitecture.md)** - 하드웨어 아키텍처
  - Pickee: 모바일 베이스, 로봇팔, LiDAR, 카메라
  - Packee: 듀얼 암, 비전 시스템
  - 서버: 메인 서버, LLM 서버

- **[SWArchitecture.md](Architecture/SWArchitecture.md)** - 소프트웨어 아키텍처
  - UI Layer: Shopee App (ROS2 지원)
  - Server Layer: Main Service, LLM Service
  - Robot Layer: Pickee/Packee 제어 시스템

### 🗄️ [ERDiagram](ERDiagram/) - 데이터베이스 설계
시스템 데이터 모델 및 관계형 데이터베이스 구조를 정의합니다.

- **[ERDiagram.md](ERDiagram/ERDiagram.md)** - 통합 ERD
  - 사용자 관리: admin, customer, allergy_info
  - 상품 관리: product, section, shelf, warehouse, location
  - 주문 관리: order, order_item, box
  - 로봇 관리: robot, robot_history

### 🔄 [StateDiagram](StateDiagram/) - 로봇 상태 설계
로봇의 동작 상태 및 전환 규칙을 정의합니다.

- **[StateDefinition.md](StateDiagram/StateDefinition.md)** - 상태 정의
  - Pickee 상태: 공통(PK_S00~S02), 쇼핑(PK_S10~S16), 직원보조(PK_S20~S23)
  - Packee 상태: 포장 작업(PA_S00~S05)

- **[StateDiagram_Pickee.md](StateDiagram/StateDiagram_Pickee.md)** - Pickee 상태 다이어그램
- **[StateDiagram_Packee.md](StateDiagram/StateDiagram_Packee.md)** - Packee 상태 다이어그램

### 📋 [SequenceDiagram](SequenceDiagram/) - 시나리오 설계
주요 사용 시나리오별 상세 시퀀스를 정의합니다.

#### 🔐 인증 및 기본 기능
- **[SC_01_1.md](SequenceDiagram/SC_01_1.md)** - 로그인 (성공)
- **[SC_01_2.md](SequenceDiagram/SC_01_2.md)** - 로그인 (실패)  
- **[SC_01_3.md](SequenceDiagram/SC_01_3.md)** - 로그아웃

#### 🛒 상품 검색 및 예약
- **[SC_02_1.md](SequenceDiagram/SC_02_1.md)** - 텍스트 상품 검색
- **[SC_02_2_1.md](SequenceDiagram/SC_02_2_1.md)** - 음성 상품 검색 (성공)
- **[SC_02_2_2.md](SequenceDiagram/SC_02_2_2.md)** - 음성 상품 검색 (실패)
- **[SC_02_3_1.md](SequenceDiagram/SC_02_3_1.md)** - 상품 예약 (성공)
- **[SC_02_3_2.md](SequenceDiagram/SC_02_3_2.md)** - 상품 예약 (실패)

#### 💳 결제 및 쇼핑 시작
- **[SC_02_4.md](SequenceDiagram/SC_02_4.md)** - 상품 결제
- **[SC_02_5.md](SequenceDiagram/SC_02_5.md)** - 원격 쇼핑 시작

#### 🤖 로봇 제어 및 모니터링
- **[SC_03_1.md](SequenceDiagram/SC_03_1.md)** - 실시간 영상 모니터링 시작
- **[SC_03_2.md](SequenceDiagram/SC_03_2.md)** - 실시간 영상 모니터링 종료
- **[SC_03_3.md](SequenceDiagram/SC_03_3.md)** - 실시간 상품 선택
- **[SC_03_4.md](SequenceDiagram/SC_03_4.md)** - 쇼핑 중 알림

#### 📦 포장 및 완료
- **[SC_04.md](SequenceDiagram/SC_04.md)** - 상품 포장

#### 🔍 관리 기능
- **[SC_05_1_1.md](SequenceDiagram/SC_05_1_1.md)** - 주문 정보 조회
- **[SC_05_2_1.md](SequenceDiagram/SC_05_2_1.md)** - 작업 정보 조회  
- **[SC_06_1.md](SequenceDiagram/SC_06_1.md)** - 로봇 상태 조회

### 🔌 [InterfaceSpecification](InterfaceSpecification/) - 인터페이스 명세
컴포넌트 간 통신 프로토콜 및 데이터 형식을 정의합니다.

#### 📱 App ↔ Main Service
- **[App_vs_Main.md](InterfaceSpecification/App_vs_Main.md)** - TCP 기반 명령/응답
- **[App_vs_Main_UDP.md](InterfaceSpecification/App_vs_Main_UDP.md)** - UDP 영상 스트리밍
- **[App_vs_RobotControllers.md](InterfaceSpecification/App_vs_RobotControllers.md)** - ROS2 모니터링

#### 🖥️ Main Service ↔ 외부 서비스
- **[Main_vs_LLM.md](InterfaceSpecification/Main_vs_LLM.md)** - HTTP REST API
- **[Main_vs_Pic_Main.md](InterfaceSpecification/Main_vs_Pic_Main.md)** - ROS2 서비스/토픽
- **[Main_vs_Pac_Main.md](InterfaceSpecification/Main_vs_Pac_Main.md)** - ROS2 서비스/토픽

#### 🤖 로봇 내부 통신
- **[Pic_Main_vs_Pic_Mobile.md](InterfaceSpecification/Pic_Main_vs_Pic_Mobile.md)** - 모바일 베이스 제어
- **[Pic_Main_vs_Pic_Arm.md](InterfaceSpecification/Pic_Main_vs_Pic_Arm.md)** - 로봇팔 제어
- **[Pic_Main_vs_Pic_Vision.md](InterfaceSpecification/Pic_Main_vs_Pic_Vision.md)** - 비전 시스템
- **[Pac_Main_vs_Pac_Arm.md](InterfaceSpecification/Pac_Main_vs_Pac_Arm.md)** - 듀얼 암 제어
- **[Pac_Main_vs_Pac_Vision.md](InterfaceSpecification/Pac_Main_vs_Pac_Vision.md)** - 포장 비전

### 🛠️ [DevelopmentPlan](DevelopmentPlan/) - 개발 계획
각 컴포넌트별 상세 설계 및 구현 계획을 제공합니다.

#### 🖥️ 서비스 계층
- **[MainService/](DevelopmentPlan/MainService/)** - 중앙 백엔드 서비스
  - [MainServiceDesign.md](DevelopmentPlan/MainService/MainServiceDesign.md) - 서비스 설계
  - [DashboardDesign.md](DevelopmentPlan/MainService/DashboardDesign.md) - 관리 대시보드
  - [robot_fleet_management.md](DevelopmentPlan/MainService/robot_fleet_management.md) - 로봇 플릿 관리

- **[LLMService/](DevelopmentPlan/LLMService/)** - 자연어 처리 서비스
  - [LLMServiceDesign.md](DevelopmentPlan/LLMService/LLMServiceDesign.md) - LLM 서비스 설계
  - [LLMServicePlan.md](DevelopmentPlan/LLMService/LLMServicePlan.md) - 개발 계획

- **[App/](DevelopmentPlan/App/)** - 사용자 인터페이스
  - [AppPlan.md](DevelopmentPlan/App/AppPlan.md) - Qt 기반 GUI 개발

#### 🤖 Pickee 로봇 (쇼핑 로봇)
- **[PickeeMain/](DevelopmentPlan/PickeeMain/)** - 메인 컨트롤러
  - [PickeeMainControllerDesign.md](DevelopmentPlan/PickeeMain/PickeeMainControllerDesign.md) - 제어 설계
  - [PickeeMainControllerPlan.md](DevelopmentPlan/PickeeMain/PickeeMainControllerPlan.md) - 개발 계획

- **[PickeeMobile/](DevelopmentPlan/PickeeMobile/)** - 모바일 베이스
  - [PickeeMobileDesign.md](DevelopmentPlan/PickeeMobile/PickeeMobileDesign.md) - 자율 주행 설계
  - [PickeeMobilePlan.md](DevelopmentPlan/PickeeMobile/PickeeMobilePlan.md) - 개발 계획

- **[PickeeVision/](DevelopmentPlan/PickeeVision/)** - 비전 시스템
  - [PickeeVisionDesign.md](DevelopmentPlan/PickeeVision/PickeeVisionDesign.md) - 컴퓨터 비전 설계
  - [PickeeVisionPlan.md](DevelopmentPlan/PickeeVision/PickeeVisionPlan.md) - 개발 계획

- **[PickeeArm/](DevelopmentPlan/PickeeArm/)** - 로봇팔 제어
  - [PickeeArmDesign.md](DevelopmentPlan/PickeeArm/PickeeArmDesign.md) - 매니퓰레이션 설계
  - [PickeeArmPlan.md](DevelopmentPlan/PickeeArm/PickeeArmPlan.md) - 개발 계획

#### 📦 Packee 로봇 (포장 로봇)
- **[PackeeMain/](DevelopmentPlan/PackeeMain/)** - 메인 컨트롤러
  - [PackeeMainDesign.md](DevelopmentPlan/PackeeMain/PackeeMainDesign.md) - 포장 제어 설계
  - [PackeeMainPlan.md](DevelopmentPlan/PackeeMain/PackeeMainPlan.md) - 개발 계획

- **[PackeeArm/](DevelopmentPlan/PackeeArm/)** - 듀얼 암 시스템
  - [PackeeArmDesign.md](DevelopmentPlan/PackeeArm/PackeeArmDesign.md) - 듀얼 암 협업 설계
  - [PackeeArmPlan.md](DevelopmentPlan/PackeeArm/PackeeArmPlan.md) - 개발 계획

- **[PackeeVision/](DevelopmentPlan/PackeeVision/)** - 포장 비전
  - [PackeeVisionDesign.md](DevelopmentPlan/PackeeVision/PackeeVisionDesign.md) - 상품 인식 설계
  - [PackeeVisionPlan.md](DevelopmentPlan/PackeeVision/PackeeVisionPlan.md) - 개발 계획

### 📏 [CodingStandard](CodingStandard/) - 코딩 표준
프로젝트 전반의 코딩 규칙 및 스타일 가이드를 정의합니다.

- **[standard.md](CodingStandard/standard.md)** - 통합 코딩 표준
  - ROS2 표준: Package/Node/Topic/Service 명명 규칙
  - Python 표준: PEP8 기반 스타일 가이드
  - C++ 표준: Google Style Guide 기반
  - 공통 규칙: 주석, 포맷팅, 문서화

## 🎯 주요 시스템 특징

### 🤖 로봇 자율 기능
- **SLAM 기반 자율 주행**: LiDAR를 활용한 실시간 맵핑 및 경로 계획
- **동적 장애물 회피**: 사람, 카트 등 실시간 장애물 감지 및 회피
- **컴퓨터 비전**: YOLOv8 기반 상품/객체 인식 및 추적
- **듀얼 암 협업**: Packee의 양팔 협업을 통한 효율적 포장

### 🌐 통합 통신 아키텍처
- **ROS2 미들웨어**: 로봇 간 실시간 통신 및 분산 제어
- **TCP/UDP 하이브리드**: 명령/응답(TCP) + 영상 스트리밍(UDP)
- **HTTP REST API**: LLM 서비스와의 자연어 처리 연동
- **실시간 모니터링**: WebSocket 기반 실시간 상태 업데이트

### 🧠 AI/ML 통합
- **자연어 처리**: LLM 기반 음성/텍스트 상품 검색
- **컴퓨터 비전**: 딥러닝 기반 상품/장애물/직원 인식
- **경로 최적화**: A* 알고리즘 기반 최적 경로 계획
- **작업 스케줄링**: 다중 로봇 협업 및 충돌 회피

## 📊 시스템 요구사항 달성 현황

### ✅ 핵심 기능 (R1 - Critical)
- 사용자 로그인 및 인증
- 상품 검색 (텍스트/음성)
- 상품 예약 및 결제
- 실시간 상품 선택
- 자율 주행 및 장애물 회피
- 상품 포장 보조

### ✅ 주요 기능 (R2 - High)
- 실시간 영상 모니터링
- 로봇 상태 실시간 조회
- 작업 이력 관리

### 🔄 보조 기능 (R3-R4)
- 쇼핑 중 알림 시스템
- 상품 추천 엔진
- 자동 충전 시스템

## 🛠️ 개발 가이드라인

### 📋 문서 작성 규칙
1. **마크다운 표준**: GitHub Flavored Markdown 사용
2. **PlantUML 다이어그램**: 아키텍처 및 시퀀스 다이어그램
3. **한국어 문서화**: 모든 설계 문서는 한국어로 작성
4. **버전 관리**: 주요 변경사항은 문서 하단에 히스토리 기록

### 🔄 문서 업데이트 프로세스
1. **요구사항 변경** → Requirements 문서 업데이트
2. **아키텍처 수정** → Architecture 문서 반영
3. **인터페이스 변경** → InterfaceSpecification 업데이트
4. **구현 계획 수정** → DevelopmentPlan 문서 갱신

### 📝 문서 검토 체크리스트
- [ ] 요구사항과의 일관성 확인
- [ ] 아키텍처 설계와의 정합성 검증
- [ ] 인터페이스 명세 완성도 점검
- [ ] 코딩 표준 준수 여부 확인

## 🔗 관련 링크

### 📚 외부 참조 문서
- [ROS2 Jazzy Documentation](https://docs.ros.org/en/jazzy/)
- [PlantUML User Guide](https://plantuml.com/guide)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [PEP 8 Python Style Guide](https://pep8.org/)

### 🏗️ 프로젝트 저장소
- [ROS2 Workspace](../shopee_ros2/) - 실제 구현 코드
- [LLM Service](../shopee_llm/) - 자연어 처리 서비스

## 📈 문서 통계

| 카테고리 | 문서 수 | 상태 |
|----------|---------|------|
| 요구사항 | 2 | ✅ 완료 |
| 아키텍처 | 2 | ✅ 완료 |
| ERD | 1 | ✅ 완료 |
| 상태 다이어그램 | 3 | ✅ 완료 |
| 시퀀스 다이어그램 | 25 | ✅ 완료 |
| 인터페이스 명세 | 12 | ✅ 완료 |
| 개발 계획 | 20+ | 🔄 진행중 |
| 코딩 표준 | 1 | ✅ 완료 |

## 👥 문서 기여자

- **시스템 아키텍트**: 전체 시스템 설계 및 아키텍처
- **요구사항 분석가**: 사용자/시스템 요구사항 정의
- **인터페이스 설계자**: 컴포넌트 간 통신 프로토콜 설계
- **로봇 엔지니어**: 로봇 제어 시스템 설계
- **AI/ML 엔지니어**: 비전 및 자연어 처리 설계

---

**📚 체계적인 설계가 성공적인 구현의 시작입니다!**

> 이 문서들은 Shopee 프로젝트의 **살아있는 설계서**입니다. 구현 과정에서 지속적으로 업데이트되며, 모든 개발자가 참조해야 하는 **단일 진실 공급원(Single Source of Truth)**입니다.
