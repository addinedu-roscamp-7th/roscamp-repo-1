@startuml Shopee_ERD

!define PK_COLOR #FFE6CC
!define FK_COLOR #E1D5E7

entity location {
  * **id** : int <<PK>>
  --
  location_x : float
  location_y : float
  aruco_marker : int
}

entity section {
  * **id** : int <<PK>>
  --
  * location_id : int <<FK>>
  section_name : varchar
}

entity shelf {
  * **id** : int <<PK>>
  --
  * location_id : int <<FK>>
  * section_id : int <<FK>>
  shelf_name : varchar
}

entity product {
  * **id** : int <<PK>>
  --
  barcode : varchar
  name : varchar
  quantity : int
  * shelf_id : int <<FK>>
  price : int
  category : varchar
  allergy_info : varchar
  is_vegan_friendly : boolean
}

entity customer {
  * **id** : varchar <<PK>>
  --
  password : varchar
  name : varchar
  allergy_info : varchar
  is_vegan : boolean
}

entity order_info {
  * **id** : int <<PK>>
  --
  * customer_id : varchar <<FK>>
  * robot_id : int <<FK>>
  start_time : datetime
  end_time : datetime
  * order_status_id : int <<FK>>
  created_at : datetime
  updated_at : datetime
}

entity order_item_info {
  * **id** : int <<PK>>
  --
  * order_info_id : int <<FK>>
  * product_id : int <<FK>>
  quantity : int
  created_at : datetime
}

entity order_status {
  * **id** : int <<PK>>
  --
  name : varchar
}

note right of order_status
  1 : paid
  2 : fail paid
  3 : returned
  4 : fail return
  5 : packaged
  6 : fail pack
  7 : packed
  8 : fail pack
end note

entity robot {
  * **id** : int <<PK>>
  --
  is_charging : boolean
  system_error_status : varchar
  * robot_type_id : varchar <<FK>>
}

entity robot_type {
  * **id** : int <<PK>>
  --
  name : varchar
}

note right of robot_type
  1 = 주행로봇 / 2 = 로봇팔
end note

entity robot_history {
  * **id** : int <<PK>>
  --
  * robot_id : int <<FK>>
  * order_info_id : int <<FK>>
  location_history : varchar
  timestamp : datetime
  failure_reason : varchar
  is_complete : boolean
}

entity packaging {
  * **id** : int <<PK>>
  --
  * order_info_id : int <<FK>>
  package_type : varchar
  package_status : varchar
  package_start_time : datetime
  package_complete_time : datetime
  created_at : datetime
}

note right of packaging
  packaging 필요한가?
  order_status에서
  포장 여부 관리 되는지?
end note

entity admin {
  * **id** : varchar <<PK>>
  --
  password : varchar
  name : varchar
}

' Relationships
location ||--o{ section : "has"
location ||--o{ shelf : "contains"
section ||--o{ shelf : "has"
shelf ||--o{ product : "stores"

customer ||--o{ order_info : "places"
robot ||--o{ order_info : "assigned to"
order_status ||--o{ order_info : "has"
order_info ||--o{ order_item_info : "contains"
product ||--o{ order_item_info : "ordered in"

robot_type ||--o{ robot : "categorizes"
robot ||--o{ robot_history : "tracks"
order_info ||--o{ robot_history : "recorded in"

order_info ||--|| packaging : "packaged as"

@enduml