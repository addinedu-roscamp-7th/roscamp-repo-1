# Shopee Integrated Development Roadmap

본 문서는 각 세부 개발 계획을 기반으로 전체 프로젝트의 단계별 우선순위와 상호 의존 관계를 기록합니다.

## Sprint 0: 공통 기반 구축
- **공통 작업**: ROS2 워크스페이스/CI 파이프라인 구성, 공용 IDL·ROS 메시지 정의 고정, ERD를 기반으로 DB 스키마 및 마이그레이션 초안 작성.
- **서비스 베이스라인**:
  - Main Service, LLM Service, App 프로젝트 스캐폴드 생성(Hello World 수준 API/UI).
  - ROS2 기반 Pickee/Packee 컨트롤러 노드 골격 생성.
- **목표**: 핵심 컴포넌트가 동일한 인터페이스 스펙을 공유하도록 초기 구조 동결.

## Sprint 1~2: 기본 서비스 및 LLM 연동
- **Main Service**: 사용자 인증, 상품/재고 API, LLM 연동(`product_search`) 완성. DB 모델 정비 및 기본 테스트 추가.
- **LLM Service**: `/search-query`, `/intent-detection` 엔드포인트 구현 및 배포. 모델 추론 지연 측정, 샘플 데이터 확보.
- **App**: 로그인·상품 검색 UI 구축, Main Service와의 TCP 통신 모듈 연동, 목 데이터 제거.
- **성과 지표**: App에서 로그인 및 상품 검색이 실제 DB와 연동되어 응답.

## Sprint 2~4: Pickee 통합
- **Pickee Mobile**: Nav2 설정, `/move_to_location` 서비스 및 위치/속도 토픽 발행.
- **Pickee Arm**: MoveIt 기반 `/move_to_pose`, `/pick_product`, `/place_product` 서비스 및 상태 토픽 발행.
- **Pickee Vision**: 상품/장애물 탐지, 장바구니 확인 서비스 구현 후 결과 토픽 제공.
- **Pickee Main**: 상태 기계 구현, Main Service·LLM·내부 컴포넌트 연동, 원격 쇼핑 워크플로우 완성.
- **Main Service**: 주문 생성→Pickee 작업 할당→실시간 이벤트 브리지 구현.
- **App**: 실시간 쇼핑 화면(영상, bbox 선택, 장바구니 갱신)과 이벤트 처리.
- **성과 지표**: App에서 주문 생성 후 Pickee가 매대 이동·상품 피킹까지 완료하고 이벤트가 반영됨.

## Sprint 4~6: Packee 및 포장 프로세스
- **Packee Vision/Arm**: 장바구니 인식·상품 탐지·3D 좌표 산출, 포장 완료 검증.
- **Main Service**: `cart_handover_complete` 수신 시 Packee 작업 요청, 포장 결과 이벤트 전송.
- **성과 지표**: Pickee → Packee로 작업이 자동 이관되고 포장 완료가 DB/이벤트에 기록됨.

## Sprint 5 이후: 모니터링·관리 기능 고도화
- **Main Service**: 관리자 API(`robot_history_search`, 알림 이벤트) 및 UDP 영상 중계.
- **App**: 관리자 대시보드, 재고/작업 이력 UI, UDP 영상 렌더링, 실시간 로봇 상태 표시.
- **성능·안정성 보강**: 비기능 요구(지연, 신뢰성, 보안) 측정 및 튜닝, 장애/재시도 흐름 설계 반영.

## 위험 요소 및 대응 전략
- **인터페이스 변경 리스크**: Sprint 0에서 스펙 동결, 변경 시 RFC 프로세스 적용.
- **LLM 품질 불확실성**: 규칙 기반 대안 로직을 Main Service에 마련, 추후 파인튜닝.
- **실시간 스트리밍**: Sprint 2~3 중 PoC 실행, 네트워크·프레임 분할 전략 검증.
- **듀얼 암 협조 제어**: Packee Arm 개발 초기부터 시뮬레이션 환경(가제토/MoveIt)을 활용해 충돌 회피 시나리오 테스트.

## 역할 제안 (예시)
- Main Service & DB: 백엔드 팀
- App & TCP/UDP 클라이언트: 프론트엔드 팀
- Pickee 로봇 (Mobile/Arm/Vision): 로보틱스 팀 A
- Packee 로봇 (Main/Arm/Vision): 로보틱스 팀 B
- LLM Service & 도메인 데이터: AI 팀

각 팀은 Sprint 종료 시 기능 데모와 테스트 결과를 공유하여 상호 의존 작업을 조율합니다.
, 듀얼 암 피킹/플레이스 인터페이스 구현.
- **Packee Main**: 상태 기계, 작업 스케줄링, Vision·Arm 연동