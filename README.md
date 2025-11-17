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

### 프로젝트 관리
#### 컨플루언스(Confluence) - 문서 관리

![confluence](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/confluence.png?raw=true)

#### 지라(Jira) - 일정 관리

![jira](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/jira.png?raw=true)

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

![로그인](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-01-01.png?raw=true)

SC-01-02: 상품 검색

![상품 검색](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-01-02.png?raw=true)

SC-01-03: 결제

![결제](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-01-03.png?raw=true)

</details>
<details>
<summary> SC02: 쇼핑</summary>
SC-02-01: 매대 이동

![매대 이동](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-02-01.png?raw=true)

SC-02-02: 장애물 회피

![장애물회피1](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-02-02_1.png?raw=true)

![장애물회피2](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-02-02_2.png?raw=true)

SC-02-03: 매대 상품 선택

![매대 상품 선택1](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-02-03_1.png?raw=true)

![매대 상품 선택2](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-02-03_2.png?raw=true)

SC-02-04: 상품 장바구니 담기

![상품 장바구니 담기](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-02-04.png?raw=true)

SC-02-05: 쇼핑 종료

![쇼핑 종료](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-02-05.png?raw=true)

</details>
<details>
<summary> SC03: 상품 포장</summary>
SC-03-01: 포장대 이동

![포장대 이동](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-03-01.png?raw=true)

SC-03-02: 장바구니 교체

![장바구니 교체](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-03-02.png?raw=true)

SC-03-03: Packee 작업 가능 확인

![작업 가능 확인](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-03-03.png?raw=true)

SC-03-04: 상품 포장

![상품 포장](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-03-04.png?raw=true)

</details>
<details>
<summary> SC04: 복귀 및 충전</summary>

![복귀 및 충전](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-04.png?raw=true)

</details>
<details>
<summary> SC05: 관리자 기능</summary>
SC-05-01: 관리자 모니터링

로봇 정보 표시

![관리자 모니터링1](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-01_1%EB%A1%9C%EB%B4%87%EC%A0%95%EB%B3%B4%ED%91%9C%EC%8B%9C.png?raw=true)

로봇 위치 표시

![관리자 모니터링2](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-01_2%EB%A1%9C%EB%B4%87%EC%9C%84%EC%B9%98%ED%91%9C%EC%8B%9C.png?raw=true)

로봇 시야 확인

![관리자 모니터링3](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-01_3%EB%A1%9C%EB%B4%87%EC%8B%9C%EC%95%BC%ED%99%95%EC%9D%B8.png?raw=true)

로봇 시야 송출 종료

![관리자 모니터링4](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-01_4%EB%A1%9C%EB%B4%87%EC%8B%9C%EC%95%BC%EC%86%A1%EC%B6%9C%EC%A2%85%EB%A3%8C.png?raw=true)

로봇 상태 조회

![관리자 모니터링5](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-01_5%EB%A1%9C%EB%B4%87%EC%83%81%ED%83%9C%EC%A1%B0%ED%9A%8C.png?raw=true)

진행율 확인

![관리자 모니터링6](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-01_6%EC%A7%84%ED%96%89%EC%9C%A8%ED%99%95%EC%9D%B8.png?raw=true)

SC-05-02: 관리자 재고 관리

재고 정보 조회

![관리자 재고 관리1](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-02_1%EC%9E%AC%EA%B3%A0%EC%A0%95%EB%B3%B4%EC%A1%B0%ED%9A%8C.png?raw=true)

재고 정보 수정

![관리자 재고 관리2](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-02_2%EC%9E%AC%EA%B3%A0%EC%A0%95%EB%B3%B4%EC%88%98%EC%A0%95.png?raw=true)

재고 정보 추가

![관리자 재고 관리3](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-02_3%EC%9E%AC%EA%B3%A0%EC%A0%95%EB%B3%B4%EC%B6%94%EA%B0%80.png?raw=true)

재고 정보 삭제

![관리자 재고 관리4](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-02_4%EC%9E%AC%EA%B3%A0%EC%A0%95%EB%B3%B4%EC%82%AD%EC%A0%9C.png?raw=true)

SC-05-03: 관리자 작업 이력 조회

![관리자 작업 이력 조회](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-05-03.png?raw=true)

</details>
<details>
<summary> SC06: 직원 보조 기능</summary>
SC-06-01: 모드 시작

![모드 시작](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-06-01.png?raw=true)

SC-06-02: 인식 및 추종

![인식 및 추종1](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-06-02_1.png?raw=true)

![인식 및 추종1](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-06-02_2.png?raw=true)

SC-06-03: 음성 명령

![음성 명령](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-06-03.png?raw=true)

SC-06-04: 목적지 이동

![목적지 이동](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-06-04.png?raw=true)

SC-06-05: 임무 완료 확인

![임무 완료 확인](https://github.com/addinedu-roscamp-7th/roscamp-repo-1/blob/dev/assets/images/SC-06-05.png?raw=true)

</details>

### ERD

![]()

### Interface Specification

<details>
<summary> TCP 통신</summary>

| Function | From | To | Message Type | Schema |
|---------|------|----|--------------|--------|
| 사용자 로그인 요청 | App | Main Service | user_login | ```json { "type": "user_login", "data": { "user_id": "string", "password": "string" } }``` |
| 사용자 로그인 응답 | Main Service | App | user_login_response | ```json { "type": "user_login_response", "result": true, "error_code": "string", "data": { "user_id": "string", "name": "string", "gender": "boolean", "age": "int", "address": "string", "allergy_info": { "nuts": "boolean", "milk": "boolean", "seafood": "boolean", "soy": "boolean", "peach": "boolean", "gluten": "boolean", "eggs": "boolean" }, "is_vegan": "boolean" }, "message": "string" }``` |
| 유저 정보 수정 요청 | App | Main Service | user_edit | ```json { "type": "user_edit", "result": true, "error_code": "string", "data": { "user_id": "string", "name": "string", "gender": "boolean", "age": "int", "address": "string", "allergy_info": { "nuts": "boolean", "milk": "boolean", "seafood": "boolean", "soy": "boolean", "peach": "boolean", "gluten": "boolean", "eggs": "boolean" }, "is_vegan": "boolean" } }``` |
| 유저 정보 수정 응답 | Main Service | App | user_edit_response | ```json { "type": "user_edit_response", "result": true, "error_code": "string", "data": { "user_id": "string", "name": "string", "gender": "boolean", "age": "int", "address": "string", "allergy_info": { "nuts": "boolean", "milk": "boolean", "seafood": "boolean", "soy": "boolean", "peach": "boolean", "gluten": "boolean", "eggs": "boolean" }, "is_vegan": "boolean" }, "message": "string" }``` |
| 전체 상품 요청 | App | Main Service | total_product | ```json { "type": "total_product", "data": { "user_id": "string" } }``` |
| 전체 상품 응답 | Main Service | App | total_product_response | ```json { "type": "total_product_response", "result": true, "error_code": "string", "data": { "products": [ { "product_id": "int", "name": "string", "price": "int", "discount_rate": "int", "category": "string", "allergy_info": { "nuts": "boolean", "milk": "boolean", "seafood": "boolean", "soy": "boolean", "peach": "boolean", "gluten": "boolean", "eggs": "boolean" }, "is_vegan_friendly": "boolean" } ], "total_count": "int" }, "message": "string" }``` |
| 상품 검색 요청 | App | Main Service | product_search | ```json { "type": "product_search", "data": { "user_id": "string", "query": "string", "filter": { "allergy_info": { "nuts": "boolean", "milk": "boolean", "seafood": "boolean", "soy": "boolean", "peach": "boolean", "gluten": "boolean", "eggs": "boolean" }, "is_vegan": "boolean" } } }``` |
| 상품 검색 응답 | Main Service | App | product_search_response | ```json { "type": "product_search_response", "result": true, "error_code": "string", "data": { "products": [ { "product_id": "int", "name": "string", "price": "int", "quantity": "int", "section_id": "int", "category": "string", "allergy_info_id": "int", "is_vegan_friendly": "boolean" } ], "total_count": "int" }, "message": "string" }``` |
| 주문 생성 요청 | App | Main Service | order_create | ```json { "type": "order_create", "data": { "user_id": "string", "cart_items": [ { "product_id": "int", "quantity": "int" } ], "payment_method": "string", "total_amount": "int" } }``` |
| 주문 생성 응답 | Main Service | App | order_create_response | ```json { "type": "order_create_response", "result": true, "error_code": "string", "data": { "order_id": "int", "robot_id": "int", "products": [ { "product_id": "int", "name": "string", "quantity": "int", "auto_select": "boolean" } ], "total_count": "int" }, "message": "string" }``` |
| 상품 선택 요청 | App | Main Service | product_selection | ```json { "type": "product_selection", "data": { "order_id": "int", "robot_id": "int", "bbox_number": "int", "product_id": "int" } }``` |
| 상품 선택 응답 | Main Service | App | product_selection_response | ```json { "type": "product_selection_response", "result": true, "error_code": "string", "data": { "order_id": "int", "product_id": "int", "bbox_number": "int" }, "message": "string" }``` |
| 상품 선택 요청 (텍스트) | App | Main Service | product_selection_by_text | ```json { "type": "product_selection_by_text", "data": { "order_id": "int", "robot_id": "int", "speech": "string" } }``` |
| 상품 선택 응답 (텍스트) | Main Service | App | product_selection_by_text_response | ```json { "type": "product_selection_by_text_response", "result": true, "error_code": "string", "data": { "bbox": "int", "product_id": "int" }, "message": "string" }``` |
| 쇼핑 종료 요청 | App | Main Service | shopping_end | ```json { "type": "shopping_end", "data": { "user_id": "string", "order_id": "int" } }``` |
| 쇼핑 종료 응답 | Main Service | App | shopping_end_response | ```json { "type": "shopping_end_response", "result": true, "error_code": "string", "data": { "order_id": "int", "total_items": "int", "total_price": "int" }, "message": "string" }``` |
| 영상 스트림 시작 요청 | App | Main Service | video_stream_start | ```json { "type": "video_stream_start", "data": { "user_type": "string", "user_id": "string", "robot_id": "int", "camera_type": "string" } }``` |
| 영상 스트림 시작 응답 | Main Service | App | video_stream_start_response | ```json { "type": "video_stream_start_response", "result": true, "error_code": "string", "data": {}, "message": "string" }``` |
| 영상 스트림 중지 요청 | App | Main Service | video_stream_stop | ```json { "type": "video_stream_stop", "data": { "user_type": "string", "user_id": "string", "robot_id": "int" } }``` |
| 영상 스트림 중지 응답 | Main Service | App | video_stream_stop_response | ```json { "type": "video_stream_stop_response", "result": true, "error_code": "string", "data": {}, "message": "string" }``` |
| 재고 조회 요청 | App | Main Service | inventory_search | ```json { "type": "inventory_search", "data": { "product_id": "int|null", "barcode": "string|null", "name": "string|null", "quantity": ["int","int"]|null, "price": "int|null", "section_id": "int|null", "category": "string|null", "allergy_info_id": "int|null", "is_vegan_friendly": "boolean|null" } }``` |
| 재고 조회 응답 | Main Service | App | inventory_search_response | ```json { "type": "inventory_search_response", "result": true, "error_code": "string", "data": { "products": [ { "product_id": "int", "barcode": "string", "name": "string", "quantity": "int", "price": "int", "section_id": "int", "category": "string", "allergy_info_id": "int", "is_vegan_friendly": "boolean" } ], "total_count": "int" }, "message": "string" }``` |
| 재고 추가 요청 | App | Main Service | inventory_create | ```json { "type": "inventory_create", "data": { "product_id": "int", "barcode": "string", "name": "string", "quantity": "int", "price": "int", "section_id": "int", "category": "string", "allergy_info_id": "int", "is_vegan_friendly": "boolean" } }``` |
| 재고 추가 응답 | Main Service | App | inventory_create_response | ```json { "type": "inventory_create_response", "result": true, "error_code": "string", "data": {}, "message": "string" }``` |
| 재고 수정 요청 | App | Main Service | inventory_update | ```json { "type": "inventory_update", "data": { "product_id": "int", "barcode": "string", "name": "string", "quantity": "int", "price": "int", "section_id": "int", "category": "string", "allergy_info_id": "int", "is_vegan_friendly": "boolean" } }``` |
| 재고 수정 응답 | Main Service | App | inventory_update_response | ```json { "type": "inventory_update_response", "result": true, "error_code": "string", "data": {}, "message": "string" }``` |
| 재고 삭제 요청 | App | Main Service | inventory_delete | ```json { "type": "inventory_delete", "data": { "product_id": "int" } }``` |
| 재고 삭제 응답 | Main Service | App | inventory_delete_response | ```json { "type": "inventory_delete_response", "result": true, "error_code": "string", "data": {}, "message": "string" }``` |
| 작업 이력 조회 요청 | App | Main Service | robot_history_search | ```json { "type": "robot_history_search", "data": { "robot_history_id": "int|null", "robot_id": "int|null", "order_item_id": "int|null", "failure_reason": "string|null", "is_complete": "boolean|null", "active_duration": "int|null", "created_at": "string|null" } }``` |
| 작업 이력 조회 응답 | Main Service | App | robot_history_search_response | ```json { "type": "robot_history_search_response", "result": true, "error_code": "string", "data": { "histories": [ { "robot_history_id": "int", "robot_id": "int", "order_item_id": "int|null", "failure_reason": "string|null", "is_complete": "boolean", "active_duration": "int", "created_at": "datetime" } ], "total_count": "int" }, "message": "string" }``` |
| 로봇 상태 조회 요청 | App | Main Service | robot_status_request | ```json { "type": "robot_status_request", "data": { "robot_type": "string|null" } }``` |
| 로봇 상태 조회 응답 | Main Service | App | robot_status_response | ```json { "type": "robot_status_response", "result": true, "error_code": "string", "data": { "robots": [ { "robot_id": "int", "type": "string", "status": "string", "detailed_status": "string", "reserved": "boolean", "active_order_id": "int|null", "battery_level": "float|null", "maintenance_mode": "boolean", "last_update": "string|null" } ], "total_count": "int" }, "message": "string" }``` |
| 로봇 유지보수 모드 설정 요청 | App | Main Service | robot_maintenance_mode | ```json { "type": "robot_maintenance_mode", "data": { "robot_id": "int", "enabled": "boolean" } }``` |
| 로봇 유지보수 모드 설정 응답 | Main Service | App | robot_maintenance_mode_response | ```json { "type": "robot_maintenance_mode_response", "result": true, "error_code": "string", "data": { "robot_id": "int", "maintenance_mode": "boolean" }, "message": "string" }``` |
| 서비스 헬스체크 요청 | App | Main Service | health_check | ```json { "type": "health_check" }``` |
| 서비스 헬스체크 응답 | Main Service | App | health_check_response | ```json { "type": "health_check_response", "result": true, "error_code": "string", "data": { "status": "string", "checks": { "database": "boolean", "ros2": "boolean", "robot_count": "int" } }, "message": "string" }``` |
| 로봇 이동 알림 | Main Service | App | robot_moving_notification | ```json { "type": "robot_moving_notification", "result": true, "error_code": "string", "data": { "order_id": "int", "robot_id": "int", "destination": "string" }, "message": "string" }``` |
| 로봇 도착 알림 | Main Service | App | robot_arrived_notification | ```json { "type": "robot_arrived_notification", "result": true, "error_code": "string", "data": { "order_id": "int", "robot_id": "int", "location_id": "int", "section_id": "int" }, "message": "string" }``` |
| 상품 담기 완료 알림 | Main Service | App | picking_complete_notification | ```json { "type": "picking_complete_notification", "result": true, "error_code": "string", "data": { "order_id": "int", "robot_id": "int" }, "message": "string" }``` |
| 상품 선택 시작 알림 | Main Service | App | product_selection_start | ```json { "type": "product_selection_start", "result": true, "error_code": "string", "data": { "order_id": "int", "robot_id": "int", "products": [ { "product_id": "int", "name": "string", "bbox_number": "int" } ] }, "message": "string" }``` |
| 장바구니 담기 알림 | Main Service | App | cart_update_notification | ```json { "type": "cart_update_notification", "result": true, "error_code": "string", "data": { "order_id": "int", "robot_id": "int", "action": "string", "product": { "product_id": "int", "name": "string", "quantity": "int", "price": "int" }, "total_items": "int", "total_price": "int" }, "message": "string" }``` |
| 작업 정보 알림 | Main Service | App | work_info_notification | ```json { "type": "work_info_notification", "result": true, "error_code": "string", "data": { "robot_id": "int", "destination": "string", "progress": "int", "active_duration": "int", "user_id": "string", "customer_name": "string", "customer_allergy_info_id": "int", "customer_is_vegan": "boolean" }, "message": "string" }``` |
| 포장 정보 알림 | Main Service | App | packing_info_notification | ```json { "type": "packing_info_notification", "result": true, "error_code": "string", "data": { "order_status": "string", "product_id": "int", "product_name": "string", "product_price": "int", "product_quantity": "int" }, "message": "string" }``` |

</details>

<details>
<summary> UDP 통신</summary>

#### 통신규약
| 항목 | 내용 |
|------|------|
| Port | 6000 |
| Protocol | UDP |
| Data Format | JSON (메타데이터) + Binary (이미지 데이터) |
| Max Packet Size | 1,600 bytes (1,400 bytes data + 200 bytes header) |
| Encoding | UTF-8 (JSON), Binary (Image) |
| Image Format | JPEG |
| Resolution | 640×480 |
| Frame Delivery Type | Chunk-based Transmission (Partial Segmentation) |

#### 패킷 구조
[ JSON Header (≈200 bytes) ] + [ Binary Image Data (max 1,400 bytes) ]

#### JSON Header 포맷
```
{
  "type": "video_frame",
  "robot_id": 1,
  "frame_id": 12345,
  "chunk_idx": 0,
  "total_chunks": 50,
  "data_size": 1400,
  "timestamp": 1730000000000,
  "width": 640,
  "height": 480,
  "format": "jpeg"
}
```
#### 인터페이스 목록

Main -> App

| Function   | From  | To   | Message Type | 상세 메시지 포맷 | 비고 |
|------------|-------|------|--------------|------------------|------|
| 영상 송출 | Main Service | App | video_frame | {<br>  "type": "video_frame",<br>  "robot_id": 1,<br>  "frame_id": 12345,<br>  "chunk_idx": 0,<br>  "total_chunks": 50,<br>  "data_size": 1400,<br>  "timestamp": 1730000000000,<br>  "width": 640,<br>  "height": 480,<br>  "format": "jpeg"<br>}<br>+ Binary Data (max 1,400 bytes) | 640x480 JPEG 분할 전송 |

Pic Vision -> Main

| Function   | From       | To   | Message Type | 상세 메시지 포맷 | 비고 |
|------------|------------|------|--------------|------------------|------|
| 영상 송출 | Pic Vision | Main | video_frame  | {<br>  "type": "video_frame",<br>  "robot_id": 1,<br>  "frame_id": 12345,<br>  "chunk_idx": 0,<br>  "total_chunks": 50,<br>  "data_size": 1400,<br>  "timestamp": 1730000000000,<br>  "width": 640,<br>  "height": 480,<br>  "format": "jpeg"<br>}<br>+ Binary Data (max 1,400 bytes) | 640x480 JPEG 분할 전송 |


</details>

<details>
<summary> HTTP 통신</summary>

Service Name: Shopee LLM Service

Clients: Shopee Main Service, Pickee Main Controller

Port: 5001

Protocol: HTTP (RESTful)


#### 상태 코드

| 코드 (status_code) | 요청 결과                          |
|--------------------|------------------------------------|
| 200                | 정상 요청, 데이터 응답 성공        |
| 400                | 잘못된 요청 (Bad Request)          |
| 401                | 정상 요청, 정보 없음 또는 응답 실패 |
| 404                | 잘못된 요청 (Not Found)            |
| 405                | 메소드가 리소스에서 허용되지 않음   |
| 500                | 서버 내부 오류                      |
| 503                | 서비스 불가                         |

#### 인터페이스 목록

| Function           | Endpoint              | Request                               | Response                                         |
|--------------------|------------------------|----------------------------------------|--------------------------------------------------|
| 상품 검색 쿼리 생성 | GET /llm/search_query | {<br>  "text": "사과 정보 알려줘"<br>} | {<br>  "sql_query": "name LIKE '%사과%'"<br>}     |
| bbox 번호 추출     | GET /llm/bbox         | {<br>  "text": "2번 집어줘"<br>}       | {<br>  "bbox": 2<br>}                             |
| 발화 의도 분석     | GET /llm/intent_detection | {<br>  "text": "피키야, XX로 이동해줘"<br>} | {<br>  "intent": "Move_place",<br>  "entities": {<br>    "place_name": "XX",<br>    "action": "move"<br>  }<br>} |


</details>

<details>
<summary> ROS2 통신</summary>

### Main <-> Pic Main

🔹 Publish / Subscribe 메시지 (Message)

| Function         | Topic                       | Message Type                                   | From     | To   | 상세 메시지 포맷 |
|------------------|------------------------------|-------------------------------------------------|----------|------|------------------|
| 이동 시작 알림     | /pickee/moving_status        | shopee_interfaces/msg/PickeeMoveStatus.msg      | Pic Main | Main | int32 robot_id<br>int32 order_id<br>int32 location_id |
| 도착 보고         | /pickee/arrival_notice       | shopee_interfaces/msg/PickeeArrival.msg         | Pic Main | Main | int32 robot_id<br>int32 order_id<br>int32 location_id<br>int32 section_id  |
| 상품 위치 인식 완료 | /pickee/product_detected     | shopee_interfaces/msg/PickeeProductDetection.msg | Pic Main | Main | int32 robot_id<br>int32 order_id<br>DetectedProduct[] products |
| 장바구니 교체 완료 | /pickee/cart_handover_complete | shopee_interfaces/msg/PickeeCartHandover.msg    | Pic Main | Main | int32 robot_id<br>int32 order_id |
| 로봇 상태 전송      | /pickee/robot_status         | shopee_interfaces/msg/PickeeRobotStatus.msg     | Pic Main | Main | int32 robot_id<br>string state<br>float32 battery_level<br>int32 current_order_id<br>float32 position_x<br>float32 position_y<br>float32 orientation_z |
| 담기 완료 보고     | /pickee/product/selection_result | shopee_interfaces/msg/PickeeProductSelection.msg | Pic Main | Main | int32 robot_id<br>int32 order_id<br>int32 product_id<br>bool success<br>int32 quantity<br>string message |
| 창고 물품 적재 완료 | /pickee/product/loaded       | shopee_interfaces/msg/PickeeProductLoaded.msg   | Pic Main | Main | int32 robot_id<br>int32 product_id<br>int32 quantity<br>bool success<br>string message |

🔹 메시지 구조 상세 (DetectedProduct 등)

DetectedProduct
- int32 product_id
- float32 confidence
- BBox bbox
- int32 bbox_number
- DetectionInfo detection_info
- Point3D position

DetectionInfo
- Point2D[] polygon
- BBox bbox_coords

Point2D
- float32 x
- float32 y

BBox
- int32 x1
- int32 y1
- int32 x2
- int32 y2

🔹 Service 호출 표

| Function         | Service Name                         | Service Type                                          | From | To      | 상세 메시지 |
|------------------|----------------------------------------|--------------------------------------------------------|------|---------|-------------|
| 작업 시작 명령     | /pickee/workflow/start_task            | shopee_interfaces/srv/PickeeWorkflowStartTask.srv      | Main | Pic Main | **Request**<br>int32 robot_id<br>int32 order_id<br>string user_id<br>ProductLocation[] product_list<br>**Response**<br>bool success<br>string message |
| 섹션 이동 명령     | /pickee/workflow/move_to_section       | shopee_interfaces/srv/PickeeWorkflowMoveToSection.srv  | Main | Pic Main | Request: robot_id, order_id, location_id, section_id<br>Response: success, message |
| 상품 인식 명령     | /pickee/product/detect                 | shopee_interfaces/srv/PickeeProductDetect.srv          | Main | Pic Main | Request: robot_id, order_id, int32[] product_ids<br>Response: success, message |
| 상품 담기 명령     | /pickee/product/process_selection      | shopee_interfaces/srv/PickeeProductProcessSelection.srv | Main | Pic Main | Request: robot_id, order_id, product_id, bbox_number<br>Response: success, message |
| 쇼핑 종료 명령     | /pickee/workflow/end_shopping          | shopee_interfaces/srv/PickeeWorkflowEndShopping.srv     | Main | Pic Main | Request: robot_id, order_id<br>Response: success, message |
| 포장대 이동 명령   | /pickee/workflow/move_to_packaging     | shopee_interfaces/srv/PickeeWorkflowMoveToPackaging.srv | Main | Pic Main | Request: robot_id, order_id, location_id<br>Response: success, message |
| 복귀 명령         | /pickee/workflow/return_to_base        | shopee_interfaces/srv/PickeeWorkflowReturnToBase.srv    | Main | Pic Main | Request: robot_id, location_id<br>Response: success, message |
| 직원으로 복귀 명령 | /pickee/workflow/return_to_staff       | shopee_interfaces/srv/PickeeWorkflowReturnToStaff.srv   | Main | Pic Main | Request: robot_id<br>Response: success, message |
| 영상 송출 시작     | /pickee/video_stream/start             | shopee_interfaces/srv/PickeeMainVideoStreamStart.srv    | Main | Pic Main | Request: user_type, user_id, robot_id, camera_type<br>Response: success, message |
| 영상 송출 중지     | /pickee/video_stream/stop              | shopee_interfaces/srv/PickeeMainVideoStreamStop.srv     | Main | Pic Main | Request: user_type, user_id, robot_id<br>Response: success, message |
| 상품 위치 조회     | /main/get_product_location             | shopee_interfaces/srv/MainGetProductLocation.srv        | Pic Main | Main | Request: product_id<br>Response: success, warehouse_id, section_id, message |
| 좌표 정보 조회     | /main/get_location_pose               | shopee_interfaces/srv/MainGetLocationPose.srv          | Pic Main | Main | Request: location_id<br>Response: Pose2D pose, success, message |
| 창고 좌표 조회     | /main/get_warehouse_pose              | shopee_interfaces/srv/MainGetWarehousePose.srv         | Pic Main | Main | Request: warehouse_id<br>Response: Pose2D pose, success, message |
| 섹션 좌표 조회     | /main/get_section_pose                | shopee_interfaces/srv/MainGetSectionPose.srv           | Pic Main | Main | Request: section_id<br>Response: Pose2D pose, success, message |

Pose2D
- float32 x
- float32 y
- float32 theta


### Pic Main <-> Pic Vision

🟦 1. 메시지(Message) 표

| Function(기능) | Topic | Message Type | From | To | 메시지 필드 |
|----------------|--------|---------------|-------|------|-------------|
| 매대 상품 인식 완료 | /pickee/vision/detection_result | shopee_interfaces/msg/PickeeVisionDetection.msg | Pic Vision | Pic Main | int32 robot_id<br>int32 order_id<br>bool success<br>DetectedProduct[] products<br>string message |
| 장바구니 상품 확인 완료 | /pickee/vision/cart_check_result | shopee_interfaces/msg/PickeeVisionCartCheck.msg | Pic Vision | Pic Main | int32 robot_id<br>int32 order_id<br>bool success<br>int32 product_id<br>bool found<br>int32 quantity<br>string message |
| 장애물 감지 알림 | /pickee/vision/obstacle_detected | shopee_interfaces/msg/PickeeVisionObstacles.msg | Pic Vision | Pic Main | int32 robot_id<br>int32 order_id<br>Obstacle[] obstacles<br>string message |
| 직원 위치 추종 정보 | /pickee/vision/staff_location | shopee_interfaces/msg/PickeeVisionStaffLocation.msg | Pic Vision | Pic Main | int32 robot_id<br>Point2D relative_position<br>float32 distance<br>bool is_tracking |
| 직원 등록 결과 | /pickee/vision/register_staff_result | shopee_interfaces/msg/PickeeVisionStaffRegister.msg | Pic Vision | Pic Main | int32 robot_id<br>bool success<br>string message |

🟦 1-1. 메시지 구조 상세

🔸 DetectedProduct 구조

DetectedProduct
- int32 product_id
- float32 confidence
- BBox bbox
- int32 bbox_number
- DetectionInfo detection_info
- Pose6D pose

DetectionInfo
- Point2D[] polygon
- BBox bbox_coords

🔸 Obstacle 구조

Obstacle

- string obstacle_type    # cart, box, product, shelf, person, other_robot, cart_moving
- Point2D position        # (m)
- float32 distance
- float32 velocity
- Vector2D direction
- BBox bbox
- float32 confidence

🔸 공통 구조체

Point2D: float32 x, float32 y

Vector2D: float32 vx, float32 vy

BBox: int32 x1, y1, x2, y2

🟩 2. 서비스(Service) 표
| Function(기능) | Service Name | Service Type | From | To | 요청/응답 |
|----------------|---------------|-----------------------------|--------|--------|-----------|
| 매대 상품 인식 요청 | /pickee/vision/detect_products | shopee_interfaces/srv/PickeeVisionDetectProducts.srv | Pic Main | Pic Vision | Request: robot_id, order_id, int32[] product_ids<br>Response: success, message |
| 장바구니 특정 상품 확인 | /pickee/vision/check_product_in_cart | shopee_interfaces/srv/PickeeVisionCheckProductInCart.srv | Pic Main | Pic Vision | Request: robot_id, order_id, product_id<br>Response: success, message |
| 장바구니 존재 확인 | /pickee/vision/check_cart_presence | shopee_interfaces/srv/PickeeVisionCheckCartPresence.srv | Pic Main | Pic Vision | Request: robot_id, order_id<br>Response: success, bool cart_present, float confidence, message |
| 영상 송출 시작 | /pickee/vision/video_stream_start | shopee_interfaces/srv/PickeeVisionVideoStreamStart.srv | Pic Main | Pic Vision | Request: user_type, user_id, robot_id, camera_type<br>Response: success, message |
| 영상 송출 중지 | /pickee/vision/video_stream_stop | shopee_interfaces/srv/PickeeVisionVideoStreamStop.srv | Pic Main | Pic Vision | Request: user_type, user_id, robot_id<br>Response: success, message |
| 직원 등록 요청 | /pickee/vision/register_staff | shopee_interfaces/srv/PickeeVisionRegisterStaff.srv | Pic Main | Pic Vision | Request: robot_id<br>Response: accepted, message |
| 직원 추종 제어 | /pickee/vision/track_staff | shopee_interfaces/srv/PickeeVisionTrackStaff.srv | Pic Main | Pic Vision | Request: robot_id, bool track<br>Response: success, message |
| Vision 모드 변경 | /pickee/vision/set_mode | shopee_interfaces/srv/PickeeVisionSetMode.srv | Pic Main | Pic Vision | Request: robot_id, string mode<br>Response: success, message |
| 음성 출력 요청 | /pickee/tts_request | shopee_interfaces/srv/PickeeTtsRequest.srv | Pic Vision | Pic Main | Request: robot_id, text_to_speak<br>Response: success, message |

🔹 Vision Mode 종류
```
idle
navigation
register_staff
detect_products
track_staff
```

### Pic Main <-> Pic Arm

#### 메시지(Message)

#### 1. 자세 변경 상태 (`/pickee/arm/pose_status`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| robot_id | int32 | 로봇 ID |
| order_id | int32 | 주문 ID |
| pose_type | string | 자세 종류 (`shelf_view`, `cart_view`, `standby`) |
| status | string | 상태 (`in_progress`, `completed`, `failed`) |
| progress | float32 | 진행률 (0.0 ~ 1.0) |
| message | string | 메시지 |

#### 2. 픽업 상태 (`/pickee/arm/pick_status`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| robot_id | int32 | 로봇 ID |
| order_id | int32 | 주문 ID |
| product_id | int32 | 상품 ID |
| arm_side | string | 팔 구분 (Pickee는 `""`) |
| status | string | 상태 (`in_progress`, `completed`, `failed`) |
| current_phase | string | 현재 단계 (`planning`, `approaching`, `grasping`, `lifting`, `done`) |
| progress | float32 | 진행률 (0.0 ~ 1.0) |
| message | string | 메시지 |

#### 3. 담기 상태 (`/pickee/arm/place_status`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| robot_id | int32 | 로봇 ID |
| order_id | int32 | 주문 ID |
| product_id | int32 | 상품 ID |
| arm_side | string | 팔 구분 (Pickee는 `""`) |
| status | string | 상태 (`in_progress`, `completed`, `failed`) |
| current_phase | string | 현재 단계 (`planning`, `moving`, `placing`, `releasing`, `done`) |
| progress | float32 | 진행률 (0.0 ~ 1.0) |
| message | string | 메시지 |

---

#### 서비스(Service)

#### 1. 자세 변경 요청 (`/pickee/arm/move_to_pose`)
| 구분 | 필드 | 타입 | 설명 |
|-------|------|------|-----|
| Request | robot_id | int32 | 로봇 ID |
|  | order_id | int32 | 주문 ID |
|  | pose_type | string | 자세 종류 (`shelf_view`, `cart_view`, `standby`) |
| Response | success | bool | 요청 성공 여부 |
|  | message | string | 메시지 |

#### 2. 상품 확인 요청 (`/pickee/arm/check_product`)
| 구분 | 필드 | 타입 | 설명 |
|-------|------|------|-----|
| Request | bbox_number | int32 | 앱 UI용 바운딩 박스 번호 |
| Response | success | bool | 요청 성공 여부 |
|  | message | string | 메시지 |

#### 3. 상품 담기 요청 (`/pickee/arm/place_product`)
| 구분 | 필드 | 타입 | 설명 |
|-------|------|------|-----|
| Request | robot_id | int32 | 로봇 ID |
|  | order_id | int32 | 주문 ID |
|  | product_id | int32 | 상품 ID |
|  | arm_side | string | 팔 구분 (Pickee는 `""`) |
|  | pose | Pose6D | 6DOF 위치 |
| Response | success | bool | 요청 성공 여부 |
|  | message | string | 메시지 |

#### 4. Pose6D (`shopee_interfaces/msg/Pose6D`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| x | float32 | X 좌표 |
| y | float32 | Y 좌표 |
| z | float32 | Z 좌표 |
| rx | float32 | 회전 X |
| ry | float32 | 회전 Y |
| rz | float32 | 회전 Z |


### Pickee Mobile ROS Interface

#### 메시지(Message)

#### 1. 위치 업데이트 (`/pickee/mobile/pose`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| robot_id | int32 | 로봇 ID |
| order_id | int32 | 주문 ID |
| current_pose | Pose2D | 현재 위치 |
| linear_velocity | float32 | 선속도 |
| angular_velocity | float32 | 각속도 |
| battery_level | float32 | 배터리 잔량 |
| status | string | 상태 (`idle`, `moving`, `stopped`, `charging`, `error`) |

#### 2. 도착 알림 (`/pickee/mobile/arrival`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| robot_id | int32 | 로봇 ID |
| order_id | int32 | 주문 ID |
| location_id | int32 | 도착 위치 ID |
| final_pose | Pose2D | 최종 위치 |
| position_error | Pose2D | 위치 오차 |
| travel_time | float32 | 이동 시간 |
| message | string | 메시지 |

#### 3. 상태 알림 (`/pickee/mobile/status`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| robot_id | int32 | 로봇 ID |
| status | string | 상태 |

#### 4. 속도 제어 (`/pickee/mobile/speed_control`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| robot_id | int32 | 로봇 ID |
| order_id | int32 | 주문 ID |
| speed_mode | string | 속도 모드 |
| target_speed | float32 | 목표 속도 |
| obstacles | Obstacle[] | 장애물 정보 |
| reason | string | 이유 |

#### 5. 도킹 - 목적지 아르코 마커 정보 전달 (`/pickee/mobile/aruco_pose`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| aruco_id | int32 | ArUco 마커 ID |
| x | float32 | X 좌표 |
| y | float32 | Y 좌표 |
| z | float32 | Z 좌표 |
| roll | float32 | Roll |
| pitch | float32 | Pitch |
| yaw | float32 | Yaw |

#### 6. 도킹 - 도킹 완료 알림 (`/pickee/mobile/docking_result`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| data | boolean | True: 성공, False: 실패 (수신 시 작업 종료) |

#### 7. 사람 트래킹 (`/pickee/mobile/person_detection`)
| 필드 | 타입 | 설명 |
|-------|------|-----|
| robot_id | int32 | 로봇 ID |
| direction | string | 사람 방향 |

---

#### 서비스(Service)

#### 1. 목적지 이동 명령 (`/pickee/mobile/move_to_location`)
| 구분 | 필드 | 타입 | 설명 |
|-------|------|------|-----|
| Request | robot_id | int32 | 로봇 ID |
|  | order_id | int32 | 주문 ID |
|  | location_id | int32 | 목적지 위치 ID |
|  | target_pose | Pose2D | 목표 위치 |
| Response | success | bool | 요청 성공 여부 |
|  | message | string | 메시지 |

#### 2. 목적지 변경 (`/pickee/mobile/update_global_path`)
| 구분 | 필드 | 타입 | 설명 |
|-------|------|------|-----|
| Request | robot_id | int32 | 로봇 ID |
|  | order_id | int32 | 주문 ID |
|  | location_id | int32 | 목표 위치 ID |
|  | target_pose | Pose2D | 목표 위치 |
| Response | success | bool | 요청 성공 여부 |
|  | message | string | 메시지 |

#### 3. 트래킹 모드 변경 (`/pickee/mobile/change_tracking_mode`)
| 구분 | 필드 | 타입 | 설명 |
|-------|------|------|-----|
| Request | robot_id | int32 | 로봇 ID |
|  | mode | string | 트래킹 모드 |
| Response | success | bool | 요청 성공 여부 |
|  | message | string | 메시지 |




</details>

### GUI

![]()


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


# 04. 에필로그
### 프로젝트 관리
#### 컨플루언스(Confluence) - 문서 관리

#### 지라(Jira) - 일정 관리

# 마무리
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
