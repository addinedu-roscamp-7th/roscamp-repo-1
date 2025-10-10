@startuml

' 엔티티 정의
entity "customer" as customer {
  * id : varchar <<PK>>
  --
  password : varchar
  name : varchar
  gender : varchar
  age : int
  address : varchar
  allergy_info_id : int <<FK>>
  is_vegan : boolean
}

entity "admin" as admin {
  * id : varchar <<PK>>
  --
  password : varchar
  name : varchar
}

entity "allergy_info" as allergy_info {
  * id : int <<PK>>
  owner_type : enum
  --
  nuts (견과류) : boolean
  milk (유제품) : boolean
  seafood (어패류) : boolean
  soy (대두/콩) : boolean
  peach (복숭아) : boolean
  gluten (밀, 글루텐) : boolean
  eggs (계란) : boolean
}

entity "order_info" as order_info {
  * id : int <<PK>>
  --
  customer_id : varchar <<FK>>
  start_time : datetime
  end_time : datetime
  order_status : enum
  failure_reason : varchar
  created_at : datetime
}

entity "order_item_info" as order_item_info {
  * id : int <<PK>>
  --
  order_info_id : int <<FK>>
  product_id : int <<FK>>
  quantity : int
  created_at : datetime
}

entity "product" as product {
  * id : int <<PK>>
  --
  barcode : varchar
  name : varchar
  quantity : int
  price : int
  discount_rate : int
  category : varchar
  allergy_info_id : int <<FK>>
  is_vegan_friendly : boolean
  section_id : int <<FK>>
  warehouse_id : int <<FK>>
}

entity "robot" as robot {
  * id : int <<PK>>
  --
  robot_type : enum
}

entity "warehouse" as warehouse {
  * id : int <<PK>>
  --
  location_id : int <<FK>>
  warehouse_name : varchar
}

entity "robot_history" as robot_history {
  * id : int <<PK>>
  --
  robot_id : int <<FK>>
  history_type : enum
  order_item_info_id : int <<FK>> (nullable)
  is_complete : boolean
  failure_reason : varchar
  active_duration : int
  created_at : datetime
}

entity "location" as location {
  * id : int <<PK>>
  --
  location_x : float
  location_y : float
  aruco_marker (space number) : int
}

entity "shelf" as shelf {
  * id : int <<PK>>
  --
  location_id : int <<FK>>
  shelf_name : varchar
}

entity "section" as section {
  * id : int <<PK>>
  --
  shelf_id : int <<FK>>
  section_name : varchar
}

' 관계 정의
customer }o--|| allergy_info : "allergy_info_id"
order_info }o--|| customer : "customer_id"
order_item_info }o--|| order_info : "order_info_id"
order_item_info }o--|| product : "product_id"
product }o--|| allergy_info : "allergy_info_id"
product }o--|| section : "section_id"
product }o--|| warehouse : "warehouse_id"
robot_history }o--|| robot : "robot_id"
robot_history }o--|| order_item_info : "order_item_info_id"
shelf }o--|| location : "location_id"
section }o--|| shelf : "shelf_id"
warehouse }o--|| location : "location_id"

' 노트 추가
note right of allergy_info
  owner_type
  "customer" / "product"
end note

note right of order_info
  order_status:
  - 결제 성공(픽업 전): PAID
  - 결제 실패: FAIL_PAID
  - 픽업 성공(포장 전): PICKED_UP
  - 픽업 실패: FAIL_PICKUP
  - 픽업 취소(반납 전): CANCELED
  - 반납 성공: RETURNED
  - 반납 실패: FAIL_RETURN
  - 포장 성공: PACKED
  - 포장 실패: FAIL_PACK
end note

note right of robot
  robot_type
  "pickee" / "packee"
end note

note right of robot_history
  history_type:
  "task" / "charge" / "error"
  
  active_duration:
  1 = 1min
end note

@enduml
