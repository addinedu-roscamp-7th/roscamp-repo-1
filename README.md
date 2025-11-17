# roscamp-repo-1 
ROS2와 AI를 활용한 자율주행 로봇개발자 부트캠프 1팀 저장소.
=======
[![Banner](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/banner.jpg?raw=true)](https://docs.google.com/presentation/d/1-Q_TZLXfFrFoZFN47uKtgcyI_h5BXLpoyHWAMogy4Dw/edit?slide=id.p#slide=id.p)
[ㄴ 클릭시 PPT 이동](https://docs.google.com/presentation/d/1-Q_TZLXfFrFoZFN47uKtgcyI_h5BXLpoyHWAMogy4Dw/edit?slide=id.p#slide=id.p)

## 주제 : 원격 쇼핑 로봇 플랫폼 [ROS2/AI/LLM/주행/로봇팔]
![예시 이미지](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/%EC%98%88%EC%8B%9C_%EC%9D%B4%EB%AF%B8%EC%A7%80.png?raw=true)

### 프로젝트 기간
![스프린트 이미지](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/%EC%8A%A4%ED%94%84%EB%A6%B0%ED%8A%B8_%EC%9D%B4%EB%AF%B8%EC%A7%80.png?raw=true)
**25.09.10 ~ 25.11.18 약 10주간 진행** <br/>
**Sprint1** : 주제 선정 / 기획 / 요구사항 정의 <br/>
**Sprint2~4** : 설계 / 기술조사 <br/>
**Sprint5** : 통신 구현 <br/>
**Sprint6~9** : 기능 구현 및 연동 테스트 <br/>
**Sprint10** : 발표 자료 <br/>

### 활용 기술
|분류|기술|
|---|---|
|**개발환경**| <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white"/> <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=Ubuntu&logoColor=white"/> <img src="https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"/> |
|**언어**| <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white"/> |
|**UI**|<img src="https://img.shields.io/badge/PyQT-28c745?style=for-the-badge&logo=PyQT&logoColor=white"/>|
|**DBMS**| <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>|
|**AI**| <img src="https://img.shields.io/badge/YOLOv8-FFBB00?style=for-the-badge&logo=YOLO&logoColor=white" alt="YOLOv8"/> <img src="https://img.shields.io/badge/YOLOv11-FF6600?style=for-the-badge&logo=YOLO&logoColor=white" alt="YOLOv11"/>  |
|**LLM**| <img src="https://img.shields.io/badge/QWEN-aae?style=for-the-badge&logo=QWEN&logoColor=white" alt="QWEN"/> <img src="https://img.shields.io/badge/Whisper-000?style=for-the-badge&logo=Whisper&logoColor=white" alt="Whisper"/> <img src="https://img.shields.io/badge/Edge_TTS-0000EE?style=for-the-badge&logo=EDGE_TTS&logoColor=white" alt="EDGE_TTS"/>|
|**자율주행**| <img src="https://img.shields.io/badge/ROS2-225?style=for-the-badge&logo=ROS2&logoColor=white" alt="ROS2"/> <img src="https://img.shields.io/badge/Slam&Nav-595?style=for-the-badge&logo=Slam&Nav&logoColor=white" alt="ST-GCN"/> |
|**협업**|<img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white"/> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/> <img src="https://img.shields.io/badge/SLACK-4A154B?style=for-the-badge&logo=slack&logoColor=white"/> <img src="https://img.shields.io/badge/Confluence-172B4D?style=for-the-badge&logo=confluence&logoColor=white"/> <img src="https://img.shields.io/badge/JIRA-0052CC?style=for-the-badge&logo=jira&logoColor=white"/>

### 목차
- [00. 팀 소개](#00-팀-소개)
- [01. 프로젝트 소개](#01-프로젝트-소개)
- [02. 프로젝트 설계](#02-프로젝트-설계)
- [03. 프로젝트 구현](#03-프로젝트-기능-구현)
- [04. 트러블 슈팅](#05-트러블-슈팅)
- [05. 에필로그](#00-에필로그)


# 00. 팀 소개
![팀 소개 이미지](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/%ED%8C%80_%EC%86%8C%EA%B0%9C_%EC%9D%B4%EB%AF%B8%EC%A7%80.jpg?raw=true)


# 01. 프로젝트 소개
### 주제 선정 배경
![주제 선정 배경](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/%EC%A3%BC%EC%A0%9C_%EC%84%A0%EC%A0%95_%EB%B0%B0%EA%B2%BD.jpg?raw=true)

원격 쇼핑 로봇을 주제로 선정한 이유 <br/>
- AI의 발전 <br/>
- 온라인 쇼핑 품질 불량 가능성 <br/>
- 인터렉티브 콘텐츠 <br/>
- 로봇 시장 성장 <br/>
- 대형 마트와 소셜 커머스 경쟁 <br/>
- 로봇과 커머스 융합 시장 <br/>
  
등의 이유가 있겠습니다.


### 사용자 요구사항 (User Requirements)
#### 고객(Customer)

| UR_ID | Name              | Description                | Required | Remarks |
|-------|--------------------|----------------------------|----------|---------|
| UR_01 | 계정 관리         | 고객 계정 정보 관리        | R        | 계정 정보: 이름, 성별, 나이, 배송 주소, 알레르기 정보, 이전 구매 내역 |
| UR_02 | 상품 탐색         | 상품 검색 및 추천          | R        | - |
| UR_03 | 원격 쇼핑         | 원격 상품 선택 및 구매     | R        | - |
| UR_04 | 원격 쇼핑 모니터링 | 실시간 쇼핑 현황 모니터링 | R        | 로봇 위치, 이동 경로 및 ETA, 작업 상태(정지/이동/집기), 장바구니 상태, 전방 카메라 영상 |

---

#### 직원(Staff)

| UR_ID | Name             | Description                              | Required | Remarks |
|-------|------------------|------------------------------------------|----------|---------|
| UR_05 | 상품 포장 보조  | 쇼핑 종료 후 상품 적재 및 정렬           | R        | 정렬 기준: 손상 가능성 있는 물품 위로, 안전성 높은 방향 |
| UR_06 | 재고 보충 보조  | 직원 요청 시 창고 상품을 매대로 자율 운송 | O        | - |

---

#### 관리자(Admin)

| UR_ID | Name              | Description                 | Required | Remarks |
|-------|-------------------|-----------------------------|----------|---------|
| UR_07 | 주문 정보 관리   | 주문 현황 확인 및 이력 조회 | R        | 주문 정보: 주문 ID, 고객 ID, 상품 목록, 주문 상태 |
| UR_08 | 작업 정보 관리   | 작업 현황 확인 및 이력 조회 | R        | 작업 정보: 작업 ID, 고객 ID, 로봇 ID, 작업 종류, 작업 상태 |
| UR_09 | 로봇 정보 관리   | 로봇 상태 확인 및 이력 조회 | R        | 로봇 상태: 위치, 장바구니 상태, 배터리·충전, 오류 상태 |
| UR_10 | 상품 정보 관리   | 상품 정보 조회 및 수정      | R        | 상품 ID, 바코드, 이름, 수량, 가격, 카테고리, 매대 위치, 알레르기/비건 여부 |
| UR_11 | 자율 복귀        | 작업 종료 후 스테이션 자동 복귀 | O    | - |
| UR_12 | 자동 충전        | 로봇이 배터리 상태 판단 후 자동 충전 | O | - |
| UR_13 | 자율 주행        | 로봇이 목표 지점까지 자율 이동 | R | - |

[ **요약** ] <br/>
![사용자 요구사항](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/%EC%82%AC%EC%9A%A9%EC%9E%90_%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD.png?raw=true)

사용자 요구사항을 크게 3가지로 요약하면, <br/>
'Shopee App (사용자 인터페이스) / Pickee (주행&상품선택) / Packee (상품포장)' 이렇게 요약할 수 있습니다.


# 02. 프로젝트 설계
### System Requirements
| SR_ID | SR_NAME  | Description         | Priority | Remark                                         |
| ----- | -------- | ------------------- | -------- | ---------------------------------------------- |
| SR_01 | 로그인      | 고객 및 관리자가 로그인하는 기능  | R1       | 세부 기능: ID/비밀번호 인증                              |
| SR_02 | 고객 정보 조회 | 고객 및 관리자가 고객 정보를 조회 | O        | 조회 가능 정보: 이름, 성별, 나이, 알레르기 정보, 비건 유무, 이전 구매 내역 |
| SR_03 | 고객 정보 수정 | 고객/관리자가 고객 정보를 수정   | O        | 수정 가능 정보: 알레르기 정보 업데이트, 비건 유무 업데이트             |
| SR_04 | 상품 검색   | 고객이 상품 정보를 검색 | R1       | 검색어 입력: 텍스트/음성, 조회 정보: 상품명/카테고리/가격/할인율/알레르기/비건 |
| SR_05 | 상품 추천   | 고객에게 상품 추천 제공 | O        | 구매 이력 기반, 알레르기/비건 고려, 인기 상품 추천                 |
| SR_06 | 상품 예약   | 고객이 상품을 예약    | R1       | 예약 상품 추가/삭제, 수량 변경, 예약 목록 조회                   |
| SR_07 | 상품 결제     | 예약 상품 선결제 기능      | R1       |                          |
| SR_08 | 실시간 상품 선택 | 비기성품을 실시간으로 보고 선택 | R1       | 과일·육류 선택, 화면 클릭(bbox 클릭) |
| SR_09 | 장바구니 조회   | 장바구니 상품 조회        | R3       |                          |
| SR_10 | 실시간 영상 모니터링 | 로봇 카메라 영상 확인      | R2       | 전방 카메라, 로봇팔 카메라             |
| SR_11 | 쇼핑중 알림      | 특정 상황 발생 시 실시간 알림 | R3       | 매대 도착, 집기 시작/완료, 장애물 감지, 오류 |
| SR_12 | 장바구니 상태 확인  | 장바구니 현황 모니터링      | R4       | 담긴 상품 목록, 수량, 총액            |
| SR_13 | 상품 포장 보조 | 쇼핑 후 상품을 포장 박스로 적재 | R1       | 포장대 이동, 장바구니 인식, 포장 순서 계획, 듀얼암 적재, 포장 완료 검증 |
| SR_14 | 재고 보충 보조 | 로봇이 직원을 추종하며 보충 작업 지원 | O        | 직원 추종, 음성 명령, 창고-매대 운반 |
| SR_15 | 주문 정보 조회 | 주문 정보 조회    | R1       | 주문 ID, 고객 ID, 상품 목록, 금액, 상태(PAID 등), 주문 시간 |
| SR_16 | 주문 이력 조회 | 주문 이력 조회    | R2       | 주문 ID/고객 ID, 주문 일시, 상품 목록, 금액, 최종 상태       |
| SR_17 | 작업 정보 모니터링 | 모든 작업 정보/상태 모니터링 | R1       | 작업 ID, 고객/로봇 ID, 작업 종류, 상태, 시간     |
| SR_18 | 작업 이력 조회   | 작업 이력 조회         | R2       | 작업 ID, 로봇/고객 ID, 상태, 실패 이유, 위치, 시간 |
| SR_19 | 로봇 상태 조회 | 로봇 상태 실시간 조회 | R1       | 로봇 ID/타입, 위치, 장바구니 상태, 배터리, 오류, 작업 ID |
| SR_20 | 로봇 이력 조회 | 로봇 상태 이력 조회  | R2       | 위치 이동, 작업 수행, 충전, 오류, 상태 변경 timestamp |
| SR_21 | 상품 정보 조회 | 상품 정보를 조회   | R3       | 상품 ID, 바코드, 이름, 카테고리, 재고, 가격, 매대 위치, 알레르기/비건  |
| SR_22 | 상품 정보 수정 | 상품 정보를 수정   | R3       | 상품 추가/삭제, 바코드/이름 수정, 재고·가격 수정, 매대 위치, 알레르기 정보 |
| SR_23 | 로봇 자동 복귀 | 작업 종료 시 자동 복귀/다음 미션 이동 | 
| SR_24 | 로봇 자동 충전 | 배터리 상태 기반 자동 충전 | R4       |        |
| SR_25 | 장애물 회피  | 경로 중 장애물 감지·회피 경로 생성 | R1       | 정적: 카트/박스, 동적: 사람/모바일 로봇 |

[ **요약** ] <br/>
![System Requirements](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/system_requirements.png?raw=true)


### 서비스 흐름 : 주간(영업중)
![서비스흐름_영업중](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/%EC%84%9C%EB%B9%84%EC%8A%A4%ED%9D%90%EB%A6%84_%EC%98%81%EC%97%85%EC%A4%91.png?raw=true)

### 서비스 흐름 : 야간(영업외)
![서비스흐름_영업후](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/%EC%84%9C%EB%B9%84%EC%8A%A4%ED%9D%90%EB%A6%84_%EC%98%81%EC%97%85%ED%9B%84.png?raw=true)

### HW Architecture
![HW Architecture](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/HW_Arc.png?raw=true)

### SW Architecture
![SW Architecture](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SW_Arc.png?raw=true)

### 상태 다이어그램
![상태 다이어그램](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/%EC%83%81%ED%83%9C_%EB%8B%A4%EC%9D%B4%EC%96%B4%EA%B7%B8%EB%9E%A8.png?raw=true)

### 시퀀스 다이어그램
<details>
<summary> SC01: 상품 주문</summary>
SC-01-01: 로그인

SC-01-02: 상품 검색

SC-01-03: 결제

</details>
<details>
<summary> SC02: 쇼핑</summary>
SC-02-01: 매대 이동

SC-02-02: 장애물 회피

SC-02-03: 매대 상품 선택

SC-02-04: 상품 장바구니 담기

SC-02-05: 쇼핑 종료

</details>
<details>
<summary> SC03: 상품 포장</summary>
SC-03-01: 포장대 이동

SC-03-02: 장바구니 교체

SC-03-03: Packee 작업 가능 확인

SC-03-04: 상품 포장

</details>
<details>
<summary> SC04: 복귀 및 충전</summary>

</details>
<details>
<summary> SC05: 관리자 기능</summary>
SC-05-01: 관리자 모니터링

SC-05-02: 관리자 재고 관리

SC-05-03: 관리자 작업 이력 조회

</details>
<details>
<summary> SC06: 직원 보조 기능</summary>
SC-06-01: 모드 시작

SC-06-02: 인식 및 추종

SC-06-03: 음성 명령

SC-06-04: 목적지 이동

SC-06-05: 임무 완료 확인

</details>
### ERD

### Interface Specification

### GUI



# 03. 프로젝트 구현
### Shopee App

### Shopee Main

### Shopee LLM 

### Pickee Main 

### Pickee Mobile 

### Pickee Vision

### Pickee Arm

### Packee Main

### Packee Vision

### Packee Arm



# 04. 트러블 슈팅
![트러블 슈팅 1]()
![트러블 슈팅 2]()
![트러블 슈팅 3]()

# 05. 에필로그
### 프로젝트 관리
#### 컨플루언스(Confluence) - 문서 관리

#### 지라(Jira) - 일정 관리

### 마무리
| 팀 | 이름 | 소감 |
|:---:|:---:|---|
| App | 김윤재 | |
| Main | 장진혁 | |
| LLM | 김재형 | |
| Pickee 주행 | 최원호 | |
| Pickee 주행 | 임어진 | |
| Pickee 상품선택 | 이승한 | |
| Pickee 상품선택 | 류혜진 | |
| Packee | 송원준 | |
| Packee | 이한수 | |
| Packee | 박대준 | |
