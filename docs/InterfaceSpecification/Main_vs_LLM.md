# LLM Service Interfaces

**Clients:** Shopee Main Service, Pickee Main Controller

> 두 클라이언트 모두 동일한 REST 엔드포인트를 사용합니다.\
> 현재 문서는 학습/프로토타입 용도로 작성되었으며 별도의 인증·레이트 리미트 정책은 정의하지 않습니다.

main -> llm 포트: 5001

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

---

## 인터페이스 목록

### 상품 검색 쿼리 생성

**Function:** 상품 검색 쿼리 생성
**Endpoint:** `GET /llm/search_query`

#### Request
```json
{
  "text": "사과 정보 알려줘"
}
```

#### Response
```json
{
  "sql_query": "name LIKE '%사과%'"
}
```

---

### bbox 번호 추출

**Function:** bbox 번호 추출
**Endpoint:** `GET /llm/bbox`

#### 요청 (Request)
```json
{
  "text": "2번 집어줘"
}
```

#### 응답 (Response)
```json
{
  "bbox": 2
}
```
---

### 발화 의도 분석

**Function:** 발화 의도
**Endpoint:** `GET /llm/intent_detection`

#### 요청 (Request)
```json
{
  "text": "피키야,xx로 이동해줘"
}
```

#### 응답 (Response)
```json
{
  "intent": "Move_place",
  "entities": {
    "place_name": "xx",
    "action": move
  }
}
```
