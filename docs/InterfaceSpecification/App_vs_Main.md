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

사용자 로그인 요청
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

사용자 로그인 응답
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

상품 검색 요청
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

상품 검색 응답
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

주문 생성 요청
- From: App → To: Main Service
- Message Type: order_create
- 상세 메시지 포맷:
    {
      "type": "order_create",
      "data": {
        "customer_id": "U12345",
        "cart_items": [
          { "product_id": "P101", "quantity": 2 },
          { "product_id": "P202", "quantity": 1 }
        ],
        "payment_method": "card",
        "total_amount": 16200
      }
    }

주문 생성 응답
- From: Main Service → To: App
- Message Type: order_create_response
- 상세 메시지 포맷:
    {
      "type": "order_create_response",
      "result": true,
      "data": {
        "order_id": "O12345",
        "assigned_pickee": "Pickee_02",
        "estimated_time": "2025-10-05T12:35:00"
      },
      "message": "Order successfully created"
    }

상품 선택 요청
- From: App → To: Main Service
- Message Type: product_selection
- 상세 메시지 포맷:
    {
      "type": "product_selection",
      "data": {
        "order_id": "string",
        "robot_id": "int",
        "bbox_number": "int",
        "product_id": "string"
      }
    }

상품 선택 응답
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

쇼핑 종료 요청
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

쇼핑 종료 응답
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

영상 스트림 시작 요청
- From: App → To: Main Service
- Message Type: video_stream_start
- 상세 메시지 포맷:
    {
      "type": "video_stream_start",
      "data": {
        "user_type": "admin",
        "customer_id": "admin01",
        "robot_id": 1
      }
    }

영상 스트림 시작 응답
- From: Main Service → To: App
- Message Type: video_stream_start_response
- 상세 메시지 포맷 (성공):
    {
      "type": "video_stream_start_response",
      "result": true,
      "message": "비디오 송출을 시작합니다."
    }
- 상세 메시지 포맷 (실패):
    {
      "type": "video_stream_start_response",
      "result": false,
      "error_code": "SYS_001",
      "message": "Invalid server"
    }

영상 스트림 중지 요청
- From: App → To: Main Service
- Message Type: video_stream_stop
- 상세 메시지 포맷:
    {
      "type": "video_stream_stop",
      "data": {
        "user_type": "admin",
        "customer_id": "admin01",
        "robot_id": 1
      }
    }

영상 스트림 중지 응답
- From: Main Service → To: App
- Message Type: video_stream_stop_response
- 상세 메시지 포맷 (성공):
    {
      "type": "video_stream_stop_response",
      "result": true,
      "message": "비디오 송출을 중지합니다."
    }
- 상세 메시지 포맷 (실패):
    {
      "type": "video_stream_stop_response",
      "result": false,
      "error_code": "SYS_001",
      "message": "Invalid server"
    }

재고 조회 요청
- From: App → To: Main Service
- Message Type: inventory_search
- 상세 메시지 포맷:
    {
      "type": "inventory_search",
      "data": {
        "product_id": "string" || null,
        "barcode": "string" || null,
        "name": "string" || null,
        "quantity": ["int","int"] || null,
        "price": "int" || null,
        "shelf_id": "string" || null,
        "category": "string" || null,
        "allergy_info": ["string"] || null,
        "is_vegan_friendly": "boolean" || null
      }
    }
- 비고: "data"는 검색 필터 역할을 합니다.

재고 조회 응답
- From: Main Service → To: App
- Message Type: inventory_search_response
- 상세 메시지 포맷 (성공):
    {
      "type": "inventory_search_response",
      "result": true,
      "data": {
        "products" : [
          {
            "product_id": "string",
            "barcode": "string",
            "name": "string",
            "quantity": "int",
            "price": "int",
            "shelf_id": "string",
            "category": "string",
            "allergy_info": ["string"],
            "is_vegan_friendly": "boolean"
          }
        ],
        "total_count": "int"
      },
      "message": "Search completed"
    }

재고 수정 요청
- From: App → To: Main Service
- Message Type: inventory_update
- 상세 메시지 포맷:
    {
      "type": "inventory_update",
      "data": {
        "product_id": "string",
        "barcode": "string",
        "name": "string",
        "quantity": "int",
        "price": "int",
        "shelf_id": "string",
        "category": "string",
        "allergy_info": ["string"],
        "is_vegan_friendly": "boolean"
      }
    }

재고 수정 응답
- From: Main Service → To: App
- Message Type: inventory_update_response
- 상세 메시지 포맷 (성공):
    {
      "type": "inventory_update_response",
      "result": true,
      "message": "재고 정보를 수정하였습니다."
    }
- 상세 메시지 포맷 (실패):
    {
      "type": "inventory_update_response",
      "result": false,
      "error_code": "SYS_001",
      "message": "Invalid server"
    }

재고 추가 요청
- From: App → To: Main Service
- Message Type: inventory_create
- 상세 메시지 포맷:
    {
      "type": "inventory_create",
      "data": {
        "product_id": "string",
        "barcode": "string",
        "name": "string",
        "quantity": "int",
        "price": "int",
        "shelf_id": "string",
        "category": "string",
        "allergy_info": ["string"],
        "is_vegan_friendly": "boolean"
      }
    }

재고 추가 응답
- From: Main Service → To: App
- Message Type: inventory_create_response
- 상세 메시지 포맷 (성공):
    {
      "type": "inventory_create_response",
      "result": true,
      "message": "재고 정보를 추가하였습니다."
    }
- 상세 메시지 포맷 (실패):
    {
      "type": "inventory_create_response",
      "result": false,
      "error_code": "SYS_001",
      "message": "Invalid server"
    }

재고 삭제 요청
- From: App → To: Main Service
- Message Type: inventory_delete
- 상세 메시지 포맷:
    {
      "type": "inventory_delete",
      "data": {
        "product_id": "string"
      }
    }

재고 삭제 응답
- From: Main Service → To: App
- Message Type: inventory_delete_response
- 상세 메시지 포맷 (성공):
    {
      "type": "inventory_delete_response",
      "result": true,
      "message": "재고 정보를 삭제하였습니다."
    }
- 상세 메시지 포맷 (실패):
    {
      "type": "inventory_delete_response",
      "result": false,
      "error_code": "SYS_001",
      "message": "Invalid server"
    }

작업 이력 조회 요청
- From: App → To: Main Service
- Message Type: robot_history_search
- 상세 메시지 포맷:
    {
      "type": "robot_history_search",
      "data": {
        "robot_history_id": "int" || null,
        "robot_id": "int" || null,
        "order_info_id": "int" || null,
        "location_history": "string" || null,
        "failure_reason": "string" || null,
        "is_complete": "int" || null,
        "active_duration": "string" || null,
        "created_at": "string" || null
      }
    }
- 비고: "data"는 검색 필터 역할을 합니다.

작업 이력 조회 응답
- From: Main Service → To: App
- Message Type: robot_history_search_response
- 상세 메시지 포맷 (성공):
    {
      "type": "robot_history_search_response",
      "result": true,
      "data": {
        "histories" : [
          {
            "robot_history_id": "int",
            "robot_id": "int",
            "order_info_id": "int",
            "location_history": "string",
            "failure_reason": "string",
            "is_complete": "int",
            "active_duration": "string",
            "created_at": "datetime"
          }
        ],
        "total_count": "int"
      },
      "message": "Search completed"
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

작업 정보 알림 (관리자)
- From: Main Service → To: App
- Message Type: work_info_notification
- 상세 메시지 포맷:
    {
      "type": "work_info_notification",
      "result": true,
      "data": {
        "robot_id": "int",
        "destination": "string",
        "progress": "int",
        "active_duration": "int",
        "customer_id": "string",
        "customer_name": "string",
        "customer_allergy_info": ["string"],
        "customer_is_vegan": "boolean"
      },
      "message": "작업 정보 업데이트"
    }
- 비고:
    - destination: order_info.order_status가 3인 row의 다음에 있는 row가 목적지
    - progress: order_info.order_status에 3과 1의 비율로 정함
    - active_duration: robot_history의 active_duration 참조

포장 정보 알림 (관리자)
- From: Main Service → To: App
- Message Type: packing_info_notification
- 상세 메시지 포맷:
    {
      "type": "packing_info_notification",
      "result": true,
      "data": {
        "order_status": "string",
        "product_id": "string",
        "product_name": "string",
        "product_price": "int",
        "product_quantity": "int"
      },
      "message": "포장 정보 업데이트"
    }
- 비고:
    - order_status: "적재 전", "적재 완료", "포장 실패", "포장 완료"로 나뉨
    - order_status의 비율을 진행율로 표시 가능
