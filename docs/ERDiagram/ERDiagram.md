@startuml

' ==========================
'  Entity Definitions
' ==========================

entity "customer" as customer {
  * customer_id : int <<PK>>
  --
  id : varchar(20)
  password : varchar(256)
  name : varchar(10)
  gender : boolean
  age : int
  address : varchar(256)
  allergy_info_id : int <<FK>>
  is_vegan : boolean
}

entity "admin" as admin {
  * admin_id : int <<PK>>
  --
  id : varchar(20)
  password : varchar(256)
  name : varchar(10)
}

entity "allergy_info" as allergy_info {
  * allergy_info_id : int <<PK>>
  --
  nuts : boolean
  milk : boolean
  seafood : boolean
  soy : boolean
  peach : boolean
  gluten : boolean
  eggs : boolean
}

entity "order" as order {
  * order_id : int <<PK>>
  --
  customer_id : int <<FK>>
  start_time : datetime
  end_time : datetime (nullable)
  order_status : tinyint
  failure_reason : varchar(50) (nullable)
  created_at : datetime
}

entity "order_item" as order_item {
  * order_item_id : int <<PK>>
  --
  order_id : int <<FK>>
  product_id : int <<FK>>
  quantity : int
  created_at : datetime
}

entity "product" as product {
  * product_id : int <<PK>>
  --
  barcode : varchar(100)
  name : varchar(10)
  quantity : int
  price : int
  discount_rate : int
  category : varchar(10)
  allergy_info_id : int <<FK>>
  is_vegan_friendly : boolean
  section_id : int <<FK>>
  warehouse_id : int <<FK>>
}

entity "robot" as robot {
  * robot_id : int <<PK>>
  --
  robot_type : tinyint
  robot_status : tinyint
}

entity "robot_history" as robot_history {
  * robot_history_id : int <<PK>>
  --
  robot_id : int <<FK>>
  history_type : tinyint
  order_item_id : int <<FK>>
  is_complete : boolean
  failure_reason : varchar(50) (nullable)
  active_duration : int
  created_at : datetime
}

entity "warehouse" as warehouse {
  * warehouse_id : int <<PK>>
  --
  location_id : int <<FK>>
  warehouse_name : varchar(50)
}

entity "location" as location {
  * location_id : int <<PK>>
  --
  location_x : float
  location_y : float
  aruco_marker : int
}

entity "shelf" as shelf {
  * shelf_id : int <<PK>>
  --
  location_id : int <<FK>>
  shelf_name : varchar(50)
}

entity "section" as section {
  * section_id : int <<PK>>
  --
  shelf_id : int <<FK>>
  section_name : varchar(50)
}

' ==========================
'  Relationships
' ==========================

customer }o--|| allergy_info : "allergy_info_id"
order }o--|| customer : "customer_id"
order_item }o--|| order : "order_id"
order_item }o--|| product : "product_id"
product }o--|| allergy_info : "allergy_info_id"
product }o--|| section : "section_id"
product }o--|| warehouse : "warehouse_id"
robot_history }o--|| robot : "robot_id"
robot_history }o--|| order_item : "order_item_id"
warehouse }o--|| location : "location_id"
shelf }o--|| location : "location_id"
section }o--|| shelf : "shelf_id"

' ==========================
'  Notes
' ==========================

note right of order
  order_status:
  1. 결제 성공(픽업 전): PAID
  2. 결제 실패: FAIL_PAID
  3. 픽업 성공(포장 전): PICKED_UP
  4. 픽업 실패: FAIL_PICKUP
  5. 픽업 취소(반납 전): CANCELED
  6. 반납 성공: RETURNED
  7. 반납 실패: FAIL_RETURN
  8. 포장 성공: PACKED
  9. 포장 실패: FAIL_PACK
end note

note right of robot
  robot_type:
  1. pickee
  2. packee

  robot_status (pickee):
  0. 충전중
  1. 상품 위치 이동
  2. 상품 인식중
  3. 상품 선택 대기중
  4. 상품 담기중
  5. 포장대 이동중
  6. 장바구니 전달 대기중
  7. 직원 등록중
  8. 직원 추종중
  9. 창고 이동중
  10. 적재 대기중
  11. 매대 이동중
  12. 하차 대기중
  13. 대기장소 이동중

  robot_status (packee):
  0. 초기화 중
  1. 작업 대기중
  2. 장바구니 확인중
  3. 상품 인식중
  4. 작업 계획중
  5. 상품 담기중
end note

note right of robot_history
  history_type:
  1. task
  3. charge
  5. error

  active_duration:
  1 = 1min
end note

note right of allergy_info
  알레르기 정보 필드:
  - nuts: 견과류 (1. 대상, 0. 비대상)
  - milk: 유제품 (1. 대상, 0. 비대상)
  - seafood: 어패류 (1. 대상, 0. 비대상)
  - soy: 대두/콩 (1. 대상, 0. 비대상)
  - peach: 복숭아 (1. 대상, 0. 비대상)
  - gluten: 밀, 글루텐 (1. 대상, 0. 비대상)
  - eggs: 계란 (1. 대상, 0. 비대상)
  
  복수 알레르기 지원 가능
end note

@enduml
