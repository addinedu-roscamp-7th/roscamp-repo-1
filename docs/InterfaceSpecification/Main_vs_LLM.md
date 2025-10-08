# Main ↔ LLM

## HTTP 상태 코드

| 코드 (status_code) | 요청 결과 |
|---|---|
| 200 | 정상 요청, 데이터 응답 성공 |
| 400 | 잘못된 요청(Bad Request) |
| 401 | 정상 요청, 정보없음 or 응답 실패 |
| 404 | 잘못된 요청(Not Found) |
| 405 | 메소드가 리소스 허용 안됨 |
| 500 | 서버 내부 오류 |
| 503 | 서비스 불가 |

## 공통 규약

### 요청 메시지
- text: string
- request_type: string

### 응답 메시지
- text: string
- answer_type: string

## 인터페이스 목록

### IF_HTTP_01: 상품 검색 쿼리 생성

**Description:** 자연어를 DB 쿼리로 변환

| 항목 | 내용 |
|---|---|
| **From** | Main Service |
| **To** | LLM Service |
| **Request Type** | search_request |
| **Answer Type** | search_answer |

#### 요청 예시
POST /llm_service/request

    {
      "text": "사과 찾아줘",
      "request_type": "search_request"
    }

#### 응답 예시
HTTP/1.1 200 OK

    {
      "text": "SELECT * FROM product WHERE name LIKE '%사과%'",
      "answer_type": "search_answer"
    }

### IF_HTTP_02: 발화 의도 분석

**Description:** 자연어 문장의 의도와 핵심 개체를 추출

| 항목 | 내용 |
|---|---|
| **From** | Main Service / Pickee Main Controller |
| **To** | LLM Service |
| **Request Type** | intent_detection |
| **Answer Type** | intent_detection_result |

#### 요청 예시
POST /llm_service/request

    {
      "text": "피키야, B상품 1개 가져다줘",
      "request_type": "intent_detection"
    }

#### 응답 예시
HTTP/1.1 200 OK

    {
      "intent": "fetch_product",
      "entities": {
        "product_name": "B상품",
        "quantity": 1
      },
      "answer_type": "intent_detection_result"
    }