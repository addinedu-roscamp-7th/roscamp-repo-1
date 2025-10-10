공통 규약

사용 포트

TCP:5000

요청 포맷

{
  "type": "message_type",
  "data": { }
}

응답 포맷

{
  "type": "message_type",
  "result": true/false,
  "error_code": "AUTH_xxx",
  "data": { },
  "message": "string"
}

에러 코드

AUTH_xxx: 인증 관련 (001: 비밀번호 오류, 002: 사용자 없음)

ORDER_xxx: 주문 관련 (001: 잘못된 주문, 002: 결제 실패)

ROBOT_xxx: 로봇 관련 (001: 가용 로봇 없음, 002: 로봇 오류)

PROD_xxx: 상품 관련 (001: 상품 없음, 002: 재고 없음)

SYS_xxx: 시스템 관련 (001: 서버 오류)

Function

From

To

Message Type

Schema

Example & Remark

요청-응답

사용자 로그인 요청

App

Main Service

user_login

{
  "type": "user_login",
  "data": {
    "customer_id": "string",
    "password": "string"
  }
}

{
  "type": "user_login",
  "data": {
    "customer_id": "customer001",
    "password": "hunter2"
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
        "gender": "male/female",
        "age": "int",
        "address": "string",
        "allergy_info": "object",
        "is_vegan": "boolean"
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

Main Service

App

user_login_response

{
  "type": "user_login_response",
  "result": true,
  "error_code": "string",
  "data": {
    "customer_id": "string",
    "name": "string"
  },
  "message": "string"
}

{
  "type": "user_login_response",
  "result": true,
  "data": {
    "customer_id": "customer001",
    "name": "홍길동"
  },
  "message": "Login successful"
}

{
  "type": "user_login_response",
  "result": false,
  "error_code": "AUTH_001",
  "data": {},
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
        "filter": {
          "allergy_info": {
            "nuts": "boolean",
            "milk": "boolean",
            "seafood": "boolean",
            "soy": "boolean",
            "peach": "boolean",
            "gluten": "boolean",
            "eggs": "boolean",
          }
          "is_vegan": "boolean"
        }
      }
    }

App

Main Service

product_search

{
  "type": "product_search",
  "data": {
    "customer_id": "string",
    "query": "string"
  }
}

{
  "type": "product_search",
  "data": {
    "customer_id": "customer001",
    "query": "사과"
  }
}

상품 검색 응답

Main Service

App

product_search_response

{
  "type": "product_search_response",
  "result": true,
  "error_code": "string",
  "data": {
    "products": [
      {
        "product_id": "int",
        "name": "string",
        "price": "int",
        "quantity": "int",
        "section_id": "int",
        "category": "string",
        "allergy_info_id": "int",
        "is_vegan_friendly": "boolean"
      },
      ...
    ],
    "total_count": "int"
  },
  "message": "string"
}

{
  "type": "product_search_response",
  "result": true,
  "data": {
    "products": [
      {
        "product_id": 1,
        "name": "청사과",
        "price": 3200,
        "quantity": 25,
        "section_id": 101,
        "category": "fruit",
        "allergy_info_id": 12,
        "is_vegan_friendly": true
      }
    ],
    "total_count": 4
  },
  "message": "Search completed"
}

주문 생성 요청

App

Main Service

order_create

{
  "type": "order_create",
  "data": {
    "customer_id": "string",
    "cart_items": [
      {
        "product_id": int,
        "quantity": "int"
      }
    ],
    "payment_method": "string",
    "total_amount": "int"
  }
}

{
  "type": "order_create",
  "data": {
    "customer_id": "customer001",
    "cart_items": [
      { "product_id": 15, "quantity": 2 },
      { "product_id": 20, "quantity": 1 }
    ],
    "payment_method": "card",
    "total_amount": 16200
  }
}

주문 생성 응답

Main Service

App

order_create_response

{
  "type": "order_create_response",
  "result": true,
  "error_code": "string",
  "data": {
    "order_id": "int",
    "robot_id": "int",
  },
  "message": "string"
}

{
  "type": "order_create_response",
  "result": true,
  "data": {
    "order_id": 15,
    "robot_id": 3,
  },
  "message": "Order successfully created"
}

상품 선택 요청

App

Main Service

product_selection

{
  "type": "product_selection",
  "data": {
    "order_id": "int",
    "robot_id": "int",
    "bbox_number": "int",
    "product_id": "int"
  }
}

{
  "type": "product_selection",
  "data": {
    "order_id": 15,
    "robot_id": 1,
    "bbox_number": 2,
    "product_id": 45
  }
}

상품 선택 응답

Main Service

App

product_selection_response

{
  "type": "product_selection_response",
  "result": true,
  "error_code": "string",
  "data": {
    "order_id": "int",
    "product_id": "int",
    "bbox_number": "int"
  },
  "message": "string"
}

{
  "type": "product_selection_response",
  "result": true,
  "data": {
    "order_id": 15,
    "product_id": 54,
    "bbox_number": 2
  },
  "message": "Product selection confirmed"
}

쇼핑 종료 요청

App

Main Service

shopping_end

{
  "type": "shopping_end",
  "data": {
    "customer_id": "string",
    "order_id": "int"
  }
}

{
  "type": "shopping_end",
  "data": {
    "customer_id": "customer001",
    "order_id": 15
  }
}

쇼핑 종료 응답

Main Service

App

shopping_end_response

{
  "type": "shopping_end_response",
  "result": true,
  "error_code": "string",
  "data": {
    "order_id": "int",
    "total_items": "int",
    "total_price": "int"
  },
  "message": "string"
}

{
  "type": "shopping_end_response",
  "result": true,
  "data": {
    "order_id": 1,
    "total_items": 7,
    "total_price": 45800
  },
  "message": "쇼핑이 종료되었습니다"
}

영상 스트림 시작 요청

​App

Main Service

video_stream_start

{
  "type": "video_stream_start",
  "data": {
    "user_type": "string",
    "customer_id": "string",
    "robot_id": "int"
  }
}

{
  "type": "video_stream_start",
  "data": {
    "user_type": "admin",
    "customer_id": "admin01",
    "robot_id": 1
  }
}

영상 스트림 시작 응답

Main Service

App

video_stream_start_response

{
  "type": "video_stream_start_response",
  "result": true,
  "error_code": "string",
  "data": {},
  "message": "string"
}

{
  "type": "video_stream_start_response",
  "result": true,
  "data": {},
  "message": "비디오 송출을 시작합니다."
}

{
  "type": "video_stream_start_response",
  "result": false,
  "error_code": "SYS_001",
  "data": {},
  "message": "Invalid server"
}

영상 스트림 중지 요청

​App

Main Service

video_stream_stop

{
  "type": "video_stream_stop",
  "data": {
    "user_type": "string",
    "customer_id": "string",
    "robot_id": "int"
  }
}

{
  "type": "video_stream_stop",
  "data": {
    "user_type": "admin",
    "customer_id": "admin01",
    "robot_id": 1
  }
}

영상 스트림 중지 응답

Main Service

App

video_stream_stop_response

{
  "type": "video_stream_stop_response",
  "result": true,
  "error_code": "string",
  "data": {},
  "message": "string"
}

{
  "type": "video_stream_stop_response",
  "result": true,
  "data": {},
  "message": "비디오 송출을 중지합니다."
}

{
  "type": "video_stream_stop_response",
  "result": false,
  "error_code": "SYS_001",
  "data": {},
  "message": "Invalid server"
}

재고 조회 요청

​App

Main Service

inventory_search

{
  "type": "inventory_search",
  "data": {
    "product_id": "int|null",
    "barcode": "string|null",
    "name": "string|null",
    "quantity": ["int", "int"]|null,
    "price": "int|null",
    "section_id": "int|null",
    "category": "string|null",
    "allergy_info_id": "int|null",
    "is_vegan_friendly": "boolean|null"
  }
}

{
  "type": "inventory_search",
  "data": {
    "product_id": null,
    "barcode": null,
    "name": "사과",
    "quantity": null,
    "price": null,
    "section_id": 101,
    "category": "fruit",
    "allergy_info_id": null,
    "is_vegan_friendly": true
  }
}

data 객체는 검색 필터 역할

재고 조회 응답

Main Service

App

inventory_search_response

{
  "type": "inventory_search_response",
  "result": true,
  "error_code": "string",
  "data": {
    "products": [
      {
        "product_id": "int",
        "barcode": "string",
        "name": "string",
        "quantity": "int",
        "price": "int",
        "section_id": "int",
        "category": "string",
        "allergy_info_id": "int",
        "is_vegan_friendly": "boolean"
      }
    ],
    "total_count": "int"
  },
  "message": "string"
}

{
  "type": "inventory_search_response",
  "result": true,
  "data": {
    "products": [
      {
        "product_id": 1,
        "barcode": "8800000000012",
        "name": "청사과",
        "quantity": 25,
        "price": 3200,
        "section_id": 101,
        "category": "fruit",
        "allergy_info_id": 12,
        "is_vegan_friendly": true
      }
    ],
    "total_count": 4
  },
  "message": "Search completed"
}

재고 추가 요청

App

Main Service

inventory_create

{
  "type": "inventory_create",
  "data": {
    "product_id": "int",
    "barcode": "string",
    "name": "string",
    "quantity": "int",
    "price": "int",
    "section_id": "int",
    "category": "string",
    "allergy_info_id": "int",
    "is_vegan_friendly": "boolean"
  }
}

{
  "type": "inventory_create",
  "data": {
    "product_id": 50,
    "barcode": "8800000001055",
    "name": "그릭요거트",
    "quantity": 12,
    "price": 4900,
    "section_id": 205,
    "category": "dairy",
    "allergy_info_id": 18,
    "is_vegan_friendly": false
  }
}

재고 추가 응답

Main Service

App

inventory_create_response

{
  "type": "inventory_create_response",
  "result": true,
  "error_code": "string",
  "data": {},
  "message": "string"
}

{
  "type": "inventory_create_response",
  "result": true,
  "data": {},
  "message": "재고 정보를 추가하였습니다."
}

{
  "type": "inventory_create_response",
  "result": false,
  "error_code": "SYS_001",
  "data": {},
  "message": "Invalid server"
}

재고 수정 요청

App

Main Service

inventory_update

{
  "type": "inventory_update",
  "data": {
    "product_id": "int",
    "barcode": "string",
    "name": "string",
    "quantity": "int",
    "price": "int",
    "section_id": "int",
    "category": "string",
    "allergy_info_id": "int",
    "is_vegan_friendly": "boolean"
  }
}

{
  "type": "inventory_update",
  "data": {
    "product_id": 1,
    "barcode": "8800000000012",
    "name": "청사과",
    "quantity": 30,
    "price": 3200,
    "section_id": 101,
    "category": "fruit",
    "allergy_info_id": 12,
    "is_vegan_friendly": true
  }
}

재고 수정 응답

Main Service

App

inventory_update_response

{
  "type": "inventory_update_response",
  "result": true,
  "error_code": "string",
  "data": {},
  "message": "string"
}

{
  "type": "inventory_update_response",
  "result": true,
  "data": {},
  "message": "재고 정보를 수정하였습니다."
}

{
  "type": "inventory_update_response",
  "result": false,
  "error_code": "SYS_001",
  "data": {},
  "message": "Invalid server"
}

재고 삭제 요청

App

Main Service

inventory_delete

{
  "type": "inventory_delete",
  "data": {
    "product_id": "int"
  }
}

{
  "type": "inventory_delete",
  "data": {
    "product_id": 20
  }
}

재고 삭제 응답

Main Service

App

inventory_delete_response

{
  "type": "inventory_delete_response",
  "result": true,
  "error_code": "string",
  "data": {},
  "message": "string"
}

{
  "type": "inventory_delete_response",
  "result": true,
  "data": {},
  "message": "재고 정보를 삭제하였습니다."
}

{
  "type": "inventory_delete_response",
  "result": false,
  "error_code": "SYS_001",
  "data": {},
  "message": "Invalid server"
}

작업 이력 조회 요청

App

Main Service

robot_history_search

{
  "type": "robot_history_search",
  "data": {
    "robot_history_id": "int|null",
    "robot_id": "int|null",
    "order_item_info_id": "int|null",
    "failure_reason": "string|null",
    "is_complete": "boolean|null",
    "active_duration": "int|null",
    "created_at": "string|null"
  }
}

{
  "type": "robot_history_search",
  "data": {
    "robot_history_id": null,
    "robot_id": 1,
    "order_item_info_id": null,
    "failure_reason": null,
    "is_complete": null,
    "active_duration": null,
    "created_at": null
  }
}

data 객체는 검색 필터 역할

작업 이력 조회 응답

Main Service

App

robot_history_search_response

{
  "type": "robot_history_search_response",
  "result": true,
  "error_code": "string",
  "data": {
    "histories": [
      {
        "robot_history_id": "int",
        "robot_id": "int",
        "order_item_info_id": "int|null",
        "failure_reason": "string|null",
        "is_complete": "boolean",
        "active_duration": "int",
        "created_at": "datetime"
      }
    ],
    "total_count": "int"
  },
  "message": "string"
}

{
  "type": "robot_history_search_response",
  "result": true,
  "data": {
    "histories": [
      {
        "robot_history_id": 1001,
        "robot_id": 1,
        "order_item_info_id": 5012,
        "failure_reason": null,
        "is_complete": true,
        "active_duration": 7,
        "created_at": "2025-10-05T03:42:00Z"
      }
    ],
    "total_count": 1
  },
  "message": "Search completed"
}

이벤트











로봇 이동 알림

Main Service

App

robot_moving_notification

{
  "type": "robot_moving_notification",
  "result": true,
  "error_code": "string",
  "data": {
    "order_id": "int",
    "robot_id": "int",
    "destination": "string"
  },
  "message": "string"
}

{
  "type": "robot_moving_notification",
  "result": true,
  "data": {
    "order_id": 45,
    "robot_id": 1,
    "destination": "SECTION_A1_01"
  },
  "message": "상품 위치로 이동 중입니다"
}

로봇 도착 알림

Main Service

App

robot_arrived_notification

{
  "type": "robot_arrived_notification",
  "result": true,
  "error_code": "string",
  "data": {
    "order_id": "int",
    "robot_id": "int",
    "location_id": "int",
    "section_id": "int"
  },
  "message": "string"
}

{
  "type": "robot_arrived_notification",
  "result": true,
  "data": {
    "order_id": 54,
    "robot_id": 1,
    "location_id": 213,
    "section_id": 101
  },
  "message": "섹션에 도착했습니다"
}

상품 선택 시작 알림

Main Service

App

product_selection_start

{
  "type": "product_selection_start",
  "result": true,
  "error_code": "string",
  "data": {
    "order_id": "int",
    "robot_id": "int",
    "products": [
      {
        "product_id": "int",
        "name": "string",
        "bbox_number": "int"
      }
    ]
  },
  "message": "string"
}

{
  "type": "product_selection_start",
  "result": true,
  "data": {
    "order_id": 213,
    "robot_id": 1,
    "products": [
      {
        "product_id": 234,
        "name": "청사과",
        "bbox_number": 1
      },
      {
        "product_id": 43,
        "name": "빨간사과",
        "bbox_number": 2
      }
    ]
  },
  "message": "상품을 선택해주세요"
}

장바구니 담기 알림

Main Service

App

cart_update_notification

{
  "type": "cart_update_notification",
  "result": true,
  "error_code": "string",
  "data": {
    "order_id": "int",
    "robot_id": "int",
    "action": "string",
    "product": {
      "product_id": "int",
      "name": "string",
      "quantity": "int",
      "price": "int"
    },
    "total_items": "int",
    "total_price": "int"
  },
  "message": "string"
}

{
  "type": "cart_update_notification",
  "result": true,
  "data": {
    "order_id": 23,
    "robot_id": 1,
    "action": "add",
    "product": {
      "product_id": 23,
      "name": "청사과",
      "quantity": 1,
      "price": 3200
    },
    "total_items": 3,
    "total_price": 8640
  },
  "message": "상품이 장바구니에 담겼습니다"
}

작업 정보 알림 (관리자)

Main Service

App

work_info_notification

{
  "type": "work_info_notification",
  "result": true,
  "error_code": "string",
  "data": {
    "robot_id": "int",
    "destination": "string",
    "progress": "int",
    "active_duration": "int",
    "customer_id": "string",
    "customer_name": "string",
    "customer_allergy_info_id": "int",
    "customer_is_vegan": "boolean"
  },
  "message": "string"
}

{
  "type": "work_info_notification",
  "result": true,
  "data": {
    "robot_id": 1,
    "destination": "PACKING_AREA_A",
    "progress": 60,
    "active_duration": 12,
    "customer_id": "customer001",
    "customer_name": "홍길동",
    "customer_allergy_info_id": 12,
    "customer_is_vegan": false
  },
  "message": "작업 정보 업데이트"
}

destination: order_info.order_status가 3인 row의 다음 row가 목적지

progress: order_info.order_status 진행 비율

active_duration: robot_history.active_duration 참조

포장 정보 알림 (관리자)

Main Service

App

packing_info_notification

{
  "type": "packing_info_notification",
  "result": true,
  "error_code": "string",
  "data": {
    "order_status": "string",
    "product_id": "int",
    "product_name": "string",
    "product_price": "int",
    "product_quantity": "int"
  },
  "message": "string"
}

{
  "type": "packing_info_notification",
  "result": true,
  "data": {
    "order_status": "PACKING",
    "product_id": 30,
    "product_name": "청사과",
    "product_price": 3200,
    "product_quantity": 1
  },
  "message": "포장 정보 업데이트"
}

order_status: ERD 정의 enum 사용 (예: PACKED, FAIL_PACK)

order_status 비율을 진행율로 표현 가능