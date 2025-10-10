@startuml
[*] --> 초기화중 : Start

초기화중 --> 작업대기중 : 초기화 완료
작업대기중 --> 장바구니확인중 : 장바구니 교체 완료
장바구니확인중 --> 상품인식중 : 장바구니 유무 확인 완료
상품인식중 --> 작업계획중 : 상품 위치 인식 완료
작업계획중 --> 상품담기중 : 작업 계획 완료
상품담기중 --> 상품인식중 : Pick & Place 완료 [다음 상품 존재]
상품담기중 --> 작업대기중 : 모든 상품 포장 완료

@enduml

@startuml
[*] --> INITIALIZING : Start

INITIALIZING --> STANDBY : 초기화 완료
STANDBY --> CHECKING_CART : 장바구니 교체 완료
CHECKING_CART --> DETECTING_PRODUCTS : 장바구니 유무 확인 완료
DETECTING_PRODUCTS --> PLANNING_TASK : 상품 위치 인식 완료
PLANNING_TASK --> PACKING_PRODUCTS : 작업 계획 완료
PACKING_PRODUCTS --> DETECTING_PRODUCTS : Pick & Place 완료\n[다음 상품 존재]
PACKING_PRODUCTS --> STANDBY : 모든 상품\n포장 완료

@enduml