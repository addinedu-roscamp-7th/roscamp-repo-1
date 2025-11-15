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
|**LLM**| <img src="https://img.shields.io/badge/QWEN-aae?style=for-the-badge&logo=QWEN&logoColor=white" alt="QWEN"/> <img src="https://img.shields.io/badge/Whisper-000?style=for-the-badge&logo=Whisper&logoColor=white" alt="Whisper"/> <img src="https://img.shields.io/badge/Edge TTS-00C?style=for-the-badge&logo=Edge TTS&logoColor=white" alt="Edge TTS"/>|
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

### 서비스 흐름 : 주간(영업중)

### 서비스 흐름 : 야간(영업외)

### System Architecture

### 시퀀스 다이어그램

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
