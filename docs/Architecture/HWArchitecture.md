```plantuml
@startuml Shopee Architecture

!define RECTANGLE_COLOR #f5f5f5
!define SERVER_COLOR #ffe6cc
!define PICKEE_COLOR #d5e8d4
!define PACKEE_COLOR #e1d5e7
!define USER_COLOR #dae8fc

package "UI Layer" {
  component [User PC] USER_COLOR
}

package "Server Layer" {
  component [Shopee Main Server] SERVER_COLOR
  component [Shopee LLM Server] SERVER_COLOR
}

package "Robot Layer" {
  package "Pickee" RECTANGLE_COLOR {
    component [Pickee\nMobile Control\nDevice] PICKEE_COLOR as PickeeMobile
    component [Pickee\nMain Control\nDevice] PICKEE_COLOR as PickeeMain
    component [Pickee\nArm Control\nDevice] PICKEE_COLOR as PickeeArm
    
    component [LiDAR] as LiDAR
    component [DC Motor x2] as DCMotor
    component [Camera] as PickeeCamera
    component [Display] as Display
    component [Speaker] as Speaker
    component [Robot Arm] as PickeeRobotArm
  }
  
  package "Packee" RECTANGLE_COLOR {
    component [Packee\nMain Control\nDevice] PACKEE_COLOR as PackeeMain
    component [Packee\nArm Control\nDevice] PACKEE_COLOR as PackeeArm
    
    component [Robot Arm] as PackeeRobotArm
    component [Camera] as PackeeCamera
  }
}

' Connections
[User PC] <--> [Shopee Main Server]
[Shopee Main Server] <--> [Shopee LLM Server]
[Shopee Main Server] <--> PickeeMain
[Shopee Main Server] <--> PackeeMain

PickeeMobile <--> PickeeMain
PickeeMain <--> PickeeArm
PackeeMain <--> PackeeArm

LiDAR --> PickeeMobile
DCMotor <-- PickeeMobile
PickeeCamera --> PickeeMain
Display <-- PickeeMain
Speaker <-- PickeeMain
PickeeRobotArm <--> PickeeArm

PackeeRobotArm <--> PackeeArm
PackeeCamera --> PackeeMain

note right of [Shopee LLM Server]
  AI Processing
end note

note bottom of Pickee
  Mobile Robot with
  Vision and Manipulation
end note

note bottom of Packee
  Packaging Robot
end note

@enduml
```