# 📡 Interface Specification  
**Component:** App ↔ Main Service  
**Port:** TCP 5000  
**Author:** 최원호  
**Last Updated:** 1 day ago  

---

## 🔹 공통 규약

### 사용 포트  
- TCP: 5000  

### 요청 포맷  
    {
      "type": "message_type",
      "data": { }
    }

### 응답 포맷  
    {
      "type": "message_type",
      "result": true/false,
      "error_code": "AUTH_xxx",
      "data": { },
      "message": "string"
    }

### 에러 코드  
- AUTH_xxx: 인증 관련 (001: 비밀번호 오류, 002: 사용자 없음)  
- ORDER_xxx: 주문 관련 (001: 잘못된 주문, 002: 결제 실패)  
- ROBOT_xxx: 로봇 관련 (001: 가용 로봇 없음, 002: 로봇 오류)  
- PROD_xxx: 상품 관련 (001: 상품 없음, 002: 재고 없음)  
- SYS_xxx: 시스템 관련 (001: 서버 오류)  

---

## 🔹 인터페이스 목록

| IF ID | Function | From | To | Message Type | Description |
|-------|-----------|------|----|---------------|-------------|
| IF-001 | 사용자 로그인 요청 | App | Main Service | user_login | 사용자 로그인 |
| IF-002 | 사용자 로그인 응답 | Main Service | App | user_login_response | 로그인 결과 반환 |
| IF-003 | 관리자 로그인 요청 | App | Main Service | admin_login | 관리자 로그인 |
| IF-004 | 관리자 로그인 응답 | Main Service | App | admin_login_response | 관리자 로그인 결과 |
| IF-005 | 로봇 배정 요청 | App | Main Service | request_robot | 결제 전 로봇 배정 |
| IF-006 | 로봇 배정 응답 | Main Service | App | request_robot_response | 로봇 배정 결과 |
| IF-007 | 결제 완료 요청 | App | Main Service | payment_complete | 결제 완료 및 주문 생성 |
| IF-008 | 결제 완료 응답 | Main Service | App | payment_complete_response | 주문 ID 반환 |
| IF-009 | 주문 생성 | App | Main Service | order_create | 주문 생성 |
| IF-010 | 주문 생성 응답 | Main Service | App | order_create_response | 주문 생성 결과 |
| IF-011 | LLM 질의 | App | Main Service | llm_query | 자연어 질의 요청 |
| IF-012 | LLM 질의 응답 | Main Service | App | llm_query_response | 질의 결과 반환 |
| IF-013 | 로봇 이동 알림 | Main Service | App | robot_moving_notification | 이동 중 상태 알림 |
| IF-014 | 로봇 도착 알림 | Main Service | App | robot_arrived_notification | 매대 도착 알림 |
| IF-015 | 상품 검색 요청 | App | Main Service | product_search | 상품 검색 요청 |
| IF-016 | 상품 검색 응답 | Main Service | App | product_search_response | 검색 결과 반환 |
| IF-017 | 상품 상세 조회 요청 | App | Main Service | product_detail | 상품 상세 정보 요청 |
| IF-018 | 상품 상세 조회 응답 | Main Service | App | product_detail_response | 상품 상세 정보 반환 |
| IF-019 | 상품 선택 시작 알림 | Main Service | App | product_selection_start | 상품 선택 화면 전환 |
| IF-020 | 상품 선택 요청 | App | Main Service | product_selection | 상품 선택 명령 |
| IF-021 | 상품 선택 응답 | Main Service | App | product_selection_response | 선택 결과 반환 |
| IF-022 | 장바구니 담기 알림 | Main Service | App | cart_update_notification | 장바구니 업데이트 |
| IF-023 | 장바구니 조회 요청 | App | Main Service | cart_view | 장바구니 내용 조회 |
| IF-024 | 장바구니 조회 응답 | Main Service | App | cart_view_response | 장바구니 내용 반환 |
| IF-025 | 장바구니 수정 요청 | App | Main Service | cart_update | 수량 변경 / 삭제 |
| IF-026 | 장바구니 수정 응답 | Main Service | App | cart_update_response | 수정 결과 |
| IF-027 | 영상 스트림 요청 | App | Main Service | video_stream_start | 실시간 영상 요청 |
| IF-028 | 영상 스트림 응답 | Main Service | App | video_stream_start_response | 스트림 URL 반환 |
| IF-029 | 로봇 상태 업데이트 | Main Service | App | robot_status_update | 로봇 상태 전송 |
| IF-030 | 쇼핑 종료 요청 | App | Main Service | shopping_end | 쇼핑 종료 |
| IF-031 | 쇼핑 종료 응답 | Main Service | App | shopping_end_response | 종료 확인 |

---

## 🔹 상세 메시지 포맷

### 사용자 로그인 요청 (user_login)
**From:** App → Main Service  
**Description:** 사용자 로그인 요청  

    {
      "type": "user_login",
      "data": {
        "customer_id": "string",
        "password": "string"
      }
    }

---

### 사용자 로그인 응답 (user_login_response)
**From:** Main Service → App  
**Description:** 로그인 결과 반환  

성공:
    {
      "type": "user_login_response",
      "result": true,
      "data": {
        "customer_id": "string",
        "name": "string"
      },
      "message": "Login successful"
    }

실패:
    {
      "type": "user_login_response",
      "result": false,
      "error_code": "AUTH_001",
      "message": "Invalid password"
    }

---

### 로봇 배정 요청 (request_robot)
**From:** App → Main Service  
**Description:** 결제 전 로봇 배정 요청  

    {
      "type": "request_robot",
      "data": {
        "customer_id": "string"
      }
    }

---

### 로봇 배정 응답 (request_robot_response)
**From:** Main Service → App  
**Description:** 로봇 배정 결과 반환  

성공:
    {
      "type": "request_robot_response",
      "result": true,
      "data": {
        "robot_id": "int"
      },
      "message": "Robot assigned"
    }

실패:
    {
      "type": "request_robot_response",
      "result": false,
      "error_code": "ROBOT_001",
      "message": "No available robots"
    }

---

### 결제 완료 요청 (payment_complete)
**From:** App → Main Service  
**Description:** 결제 완료 및 주문 생성 요청  

    {
      "type": "payment_complete",
      "data": {
        "customer_id": "string",
        "robot_id": "int",
        "shopping_list": [
          {
            "product_id": "string",
            "quantity": "int"
          }
        ]
      }
    }

---

### 결제 완료 응답 (payment_complete_response)
**From:** Main Service → App  
**Description:** 결제 완료 결과  

    {
      "type": "payment_complete_response",
      "result": true,
      "data": {
        "order_id": "string",
        "robot_id": "int"
      },
      "message": "Payment successful"
    }

---

### 상품 검색 요청 (product_search)
**From:** App → Main Service  
**Description:** 상품 검색 (음성 / 텍스트)  

    {
      "type": "product_search",
      "data": {
        "customer_id": "string",
        "query": "string",
        "input_type": "text"
      }
    }

---

### 상품 검색 응답 (product_search_response)
**From:** Main Service → App  
**Description:** 검색 결과 반환  

    {
      "type": "product_search_response",
      "result": true,
      "data": {
        "products": [
          {
            "product_id": "string",
            "name": "string",
            "price": "int",
            "quantity": "int",
            "shelf_id": "string",
            "category": "string",
            "allergy_info": "string",
            "is_vegan": "boolean"
          }
        ],
        "total_count": "int"
      },
      "message": "Search completed"
    }

---

### 쇼핑 종료 요청 (shopping_end)
**From:** App → Main Service  
**Description:** 쇼핑 종료 요청  

    {
      "type": "shopping_end",
      "data": {
        "customer_id": "string",
        "order_id": "string"
      }
    }

---

### 쇼핑 종료 응답 (shopping_end_response)
**From:** Main Service → App  
**Description:** 쇼핑 종료 결과  

    {
      "type": "shopping_end_response",
      "result": true,
      "data": {
        "order_id": "string",
        "total_items": "int",
        "total_price": "int"
      },
      "message": "쇼핑이 종료되었습니다"
    }

---

### 주문 생성 요청 (order_create)
**From:** App → Main Service
**Description:** 주문 생성 요청

    {
      "type": "order_create",
      "data": {
        "customer_id": "string",
        "shopping_list": [
          {
            "product_id": "string",
            "quantity": "int"
          }
        ]
      }
    }

---

### 주문 생성 응답 (order_create_response)
**From:** Main Service → App
**Description:** 주문 생성 결과 반환

성공:
    {
      "type": "order_create_response",
      "result": true,
      "data": {
        "session_id": "string",
        "order_id": "string",
        "robot_id": "int"
      },
      "message": "Order created successfully"
    }

실패:
    {
      "type": "order_create_response",
      "result": false,
      "error_code": "ROBOT_001",
      "message": "No available robots"
    }

---
