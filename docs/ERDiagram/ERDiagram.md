@startuml

' ==========================
'  Entity Definitions
' ==========================

entity "admin" {
  * admin_id : int <<PK>>
  --
  user_id : VARCHAR(20)
  password : VARCHAR(256)
  name : VARCHAR(10)
}

entity "customer" {
  * customer_id : int <<PK>>
  --
  user_id : VARCHAR(20)
  password : VARCHAR(256)
  name : VARCHAR(10)
  gender : bool
  age : int
  address : VARCHAR(256)
  allergy_info_id : int <<FK>>
  is_vegan : bool
}

entity "allergy_info" {
  * allergy_info_id : int <<PK>>
  --
  nuts : bool
  milk : bool
  seafood : bool
  soy : bool
  peach : bool
  gluten : bool
  eggs : bool
}

entity "product" {
  * product_id : int <<PK>>
  --
  barcode : VARCHAR(100)
  name : VARCHAR(50)
  quantity : int
  price : int
  discount_rate : int
  category : VARCHAR(10)
  allergy_info_id : int <<FK>>
  is_vegan_friendly : bool
  section_id : int <<FK>>
  warehouse_id : int <<FK>>
  length : int
  width : int
  height : int
  weight : int
  fragile : bool
}

entity "order" {
  * order_id : int <<PK>>
  --
  customer_id : int <<FK>>
  start_time : datetime
  end_time : datetime (nullable)
  order_status : tinyint
  failure_reason : VARCHAR(50) (nullable)
  created_at : datetime
  box : int <<FK>>
}

entity "box" {
  * box_id : int <<PK>>
  --
  width : int
  depth : int
  height : int
}

entity "order_item" {
  * order_item_id : int <<PK>>
  --
  order_id : int <<FK>>
  product_id : int <<FK>>
  quantity : int
  created_at : datetime
}

entity "robot" {
  * robot_id : int <<PK>>
  --
  robot_type : tinyint
}

entity "robot_history" {
  * robot_history_id : int <<PK>>
  --
  robot_id : int <<FK>>
  history_type : tinyint
  order_item_id : int <<FK>>
  is_complete : bool
  failure_reason : VARCHAR(50) (nullable)
  active_duration : int
  created_at : datetime
}

entity "location" {
  * location_id : int <<PK>>
  --
  location_name : VARCHAR(50) (nullable)
  location_x : float
  location_y : float
  location_theta : float
  aruco_marker : int
}

entity "section" {
  * section_id : int <<PK>>
  --
  shelf_id : int <<FK>>
  location_id : int <<FK>>
  section_name : VARCHAR(50)
}

entity "shelf" {
  * shelf_id : int <<PK>>
  --
  location_id : int <<FK>>
  shelf_name : VARCHAR(50)
}

entity "warehouse" {
  * warehouse_id : int <<PK>>
  --
  location_id : int <<FK>>
  warehouse_name : VARCHAR(50)
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
section }o--|| location : "location_id"


' ==========================
'  Notes
' ==========================

note right of order {
  order_status:
  1. PAID
  2. FAIL_PAID
  3. PICKED_UP
  4. FAIL_PICKUP
  5. CANCELED
  6. RETURNED
  7. FAIL_RETURN
  8. PACKED
  9. FAIL_PACK
}

note right of robot {
  robot_type:
  1. pickee
  2. packee
}

note right of robot_history {
  history_type:
  1. task
  3. charge
  5. error
}

@enduml