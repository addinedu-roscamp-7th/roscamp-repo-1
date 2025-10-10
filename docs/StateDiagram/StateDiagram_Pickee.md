@startuml
[*] --> 초기화중 : Start

state 초기화중 {
}
초기화중 --> 충전중_작업불가 : 초기화 완료
충전중_작업불가 --> 충전중_작업가능 : 배터리 30% 진입
충전중_작업가능 --> 상품위치이동중 : 쇼핑 시작
충전중_작업가능 --> 직원등록중 : 재고 보충 모드 시작

상품위치이동중 --> 상품인식중 : 상품 인식 시작
상품인식중 --> 상품담기중 : 자동 선택 완료
상품인식중 --> 상품선택대기중 : 수동 선택 시작
상품선택대기중 --> 상품담기중 : 수동 선택 완료
상품담기중 --> 포장대이동중 : 원격쇼핑 종료
상품담기중 --> 상품위치이동중 : 다음 매대로 이동 시작

포장대이동중 --> 장바구니전달대기중 : 포장대 도착 완료
장바구니전달대기중 --> 대기장소이동중 : 장바구니 전달 완료
대기장소이동중 --> 충전중_작업불가 : 대기장소 도착
대기장소이동중 --> 직원등록중 : 재고 보충 모드 시작
대기장소이동중 --> 상품위치이동중 : 쇼핑 시작

직원등록중 --> 직원추종중 : 직원 등록 완료
직원추종중 --> 창고이동중 : 상품 가져오기 요청
창고이동중 --> 적재대기중 : 창고 도착 완료
적재대기중 --> 매대이동중 : 적재 완료
매대이동중 --> 하차대기중 : 매대 도착 완료
하차대기중 --> 직원추종중 : 하차 완료, 추가 요청 대기
하차대기중 --> 대기장소이동중 : 상품 전달 완료

@enduml


@startuml
[*] --> INITIALIZING : Start

INITIALIZING --> CHARGING_UNAVAILABLE : 초기화 완료
CHARGING_UNAVAILABLE --> CHARGING_AVAILABLE : 배터리 30% 진입

CHARGING_AVAILABLE --> MOVING_TO_SHELF : 쇼핑 시작
CHARGING_AVAILABLE --> REGISTERING_STAFF : 매대보충 시작

MOVING_TO_SHELF --> DETECTING_PRODUCT : 상품 인식 시작
DETECTING_PRODUCT --> WAITING_SELECTION : 수동 선택\n시작
DETECTING_PRODUCT --> PICKING_PRODUCT : 자동 선택\n완료
DETECTING_PRODUCT --> CHARGING_AVAILABLE : 다음 매대로\n이동 시작

WAITING_SELECTION --> PICKING_PRODUCT : 수동 선택\n완료
PICKING_PRODUCT --> MOVING_TO_PACKING : 원격쇼핑 종료
MOVING_TO_PACKING --> WAITING_HANDOVER : 포장대 도착 완료
WAITING_HANDOVER --> MOVING_TO_STANDBY : 장바구니 전달\n완료

REGISTERING_STAFF --> FOLLOWING_STAFF : 직원 등록 완료
FOLLOWING_STAFF --> MOVING_TO_WAREHOUSE : 상품 가져오기 요청
MOVING_TO_WAREHOUSE --> WAITING_LOADING : 창고 도착 완료
WAITING_LOADING --> MOVING_TO_SHELF : 적재 완료
MOVING_TO_SHELF --> WAITING_UNLOADING : 매대 도착 완료

WAITING_UNLOADING --> MOVING_TO_STANDBY : 상품 전달\n완료
WAITING_UNLOADING --> FOLLOWING_STAFF : 하차 완료, 추가 요청 대기

MOVING_TO_STANDBY --> CHARGING_UNAVAILABLE : 대기장소 도착

@enduml