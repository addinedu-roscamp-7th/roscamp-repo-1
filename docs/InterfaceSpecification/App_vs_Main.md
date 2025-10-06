# 📡 Interface Specification  
**Component:** App ↔ Main Service  
**Port:** TCP 5000  

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

## 🔹 인터페이스 상세 명세

### 요청-응답

IF-001 사용자 로그인 요청
- From: App → To: Main Service
- Message Type: user_login
- 상세 메시지 포맷:
    {
      "type": "user_login",
      "data": {
        "customer_id": "string",
        "password": "string"
      }
    }

IF-001 사용자 로그인 응답
- From: Main Service → To: App
- Message Type: user_login_response
- 상세 메시지 포맷 (성공):
    {
      "type": "user_login_response",
      "result": true,
      "data": {
        "customer_id": "string",
        "name": "string"
      },
      "message": "Login successful"
    }
- 상세 메시지 포맷 (실패):
    {
      "type": "user_login_response",
      "result": false,
      "error_code": "AUTH_001",
      "message": "Invalid password"
    }

IF-002 상품 검색 요청
- From: App → To: Main Service
- Message Type: product_search
- 상세 메시지 포맷:
    {
      "type": "product_search",
      "data": {
        "customer_id": "string",
        "query": "string",
        "input_type": "text"
      }
    }
- 비고: input_type 값은 "text" 또는 "voice"

IF-002 상품 검색 응답
- From: Main Service → To: App
- Message Type: product_search_response
- 상세 메시지 포맷:
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
            "shelf_id": "string"
          }
        ],
        "total_count": "int"
      },
      "message": "Search completed"
    }

IF-003 주문 생성 요청
- From: App → To: Main Service
- Message Type: order_create
- 상세 메시지 포맷:
    {
      "type": "order_create",
      "data": {
        "user_id": "U12345",
        "cart_items": [
          { "product_id": "P101", "quantity": 2, "price": 4500 },
          { "product_id": "P202", "quantity": 1, "price": 7200 }
        ],
        "payment_method": "card",
        "total_amount": 16200
      }
    }

IF-003 주문 생성 응답
- From: Main Service → To: App
- Message Type: order_create_response
- 상세 메시지 포맷:
    {
      "type": "order_create_response",
      "result": true,
      "error_code": "ORDER_000",
      "data": {
        "order_id": "O12345",
        "assigned_pickee": "Pickee_02",
        "estimated_time": "2025-10-05T12:35:00"
      },
      "message": "Order successfully created"
    }

IF-004 상품 선택 요청
- From: App → To: Main Service
- Message Type: product_selection
- 상세 메시지 포맷 (bbox 클릭):
    {
      "type": "product_selection",
      "data": {
        "order_id": "string",
        "robot_id": "int",
        "selection_type": "bbox",
        "bbox_number": "int",
        "product_id": "string"
      }
    }
- 상세 메시지 포맷 (음성/채팅):
    {
      "type": "product_selection",
      "data": {
        "order_id": "string",
        "robot_id": "int",
        "selection_type": "voice",
        "text": "1번 오렌지 담아줘",
        "bbox_number": "int"
      }
    }
- 비고: selection_type 값은 "bbox", "voice", "chat"

IF-004 상품 선택 응답
- From: Main Service → To: App
- Message Type: product_selection_response
- 상세 메시지 포맷:
    {
      "type": "product_selection_response",
      "result": true,
      "data": {
        "order_id": "string",
        "product_id": "string",
        "bbox_number": "int"
      },
      "message": "Product selection confirmed"
    }

IF-005 쇼핑 종료 요청
- From: App → To: Main Service
- Message Type: shopping_end
- 상세 메시지 포맷:
    {
      "type": "shopping_end",
      "data": {
        "customer_id": "string",
        "order_id": "string"
      }
    }

IF-005 쇼핑 종료 응답
- From: Main Service → To: App
- Message Type: shopping_end_response
- 상세 메시지 포맷:
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

### 이벤트

로봇 이동 알림
- From: Main Service → To: App
- Message Type: robot_moving_notification
- 상세 메시지 포맷:
    {
      "type": "robot_moving_notification",
      "result": true,
      "data": {
        "order_id": "string",
        "robot_id": "int",
        "destination": "string"
      },
      "message": "상품 위치로 이동 중입니다"
    }

로봇 도착 알림
- From: Main Service → To: App
- Message Type: robot_arrived_notification
- 상세 메시지 포맷:
    {
      "type": "robot_arrived_notification",
      "result": true,
      "data": {
        "order_id": "string",
        "robot_id": "int",
        "location_id": "string",
        "shelf_name": "string"
      },
      "message": "매대에 도착했습니다"
    }

상품 선택 시작 알림
- From: Main Service → To: App
- Message Type: product_selection_start
- 상세 메시지 포맷:
    {
      "type": "product_selection_start",
      "result": true,
      "data": {
        "order_id": "string",
        "robot_id": "int",
        "products": [
          {
            "product_id": "string",
            "name": "string",
            "bbox_number": "int"
          }
        ]
      },
      "message": "상품을 선택해주세요"
    }

장바구니 담기 알림
- From: Main Service → To: App
- Message Type: cart_update_notification
- 상세 메시지 포맷:
    {
      "type": "cart_update_notification",
      "result": true,
      "data": {
        "order_id": "string",
        "robot_id": "int",
        "action": "add",
        "product": {
          "product_id": "string",
          "name": "string",
          "quantity": "int",
          "price": "int"
        },
        "total_items": "int",
        "total_price": "int"
      },
      "message": "상품이 장바구니에 담겼습니다"
    }
