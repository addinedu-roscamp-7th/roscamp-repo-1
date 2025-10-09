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

---

## 인터페이스 목록

### 상품 검색 쿼리 생성

**Description:** 자연어를 DB 쿼리로 변환
**Endpoint:** `POST /llm_service/search-query`

#### 요청 (Request)
```json
{
  "text": "사과 찾아줘"
}
```

#### 응답 (Response)
```json
{
  "sql_query": "SELECT * FROM product WHERE name LIKE '%사과%'"
}
```

---

### 발화 의도 분석

**Description:** 자연어 문장의 의도와 핵심 개체를 추출
**Endpoint:** `POST /llm_service/intent-detection`

#### 요청 (Request)
```json
{
  "text": "피키야, B상품 1개 가져다줘"
}
```

#### 응답 (Response)
```json
{
  "intent": "fetch_product",
  "entities": {
    "product_name": "B상품",
    "quantity": 1
  }
}
```
