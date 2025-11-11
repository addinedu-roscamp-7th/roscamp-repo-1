from abc import ABC, abstractmethod

class State(ABC):
    # 상태의 기반이 되는 추상 기본 클래스
    def __init__(self, node):
        self._node = node

    @abstractmethod
    def on_enter(self):
        # 상태 진입 시 호출될 메소드
        pass

    @abstractmethod
    def execute(self):
        # 상태가 활성화되어 있는 동안 주기적으로 호출될 메소드
        pass

    @abstractmethod
    def on_exit(self):
        # 상태 이탈 시 호출될 메소드
        pass
    
    # 내부 컴포넌트 Service Client 래퍼 함수들에 접근하기 위한 헬퍼 메소드들
    async def mobile_move_to_location(self, location_id, target_pose, global_path=None, navigation_mode='normal'):
        '''
        Mobile 이동 명령 (docs 중앙집중식 설계 적용)
        
        인터페이스 명세 (Pic_Main_vs_Pic_Mobile.md) 반영:
        - 목적지만 전달하면 Mobile이 A* 알고리즘으로 Global Path 자동 생성
        - Vision 장애물 정보는 별도 토픽으로 실시간 전달됨
        - Mobile에서 통합적으로 경로 계획 + 장애물 회피 + 실행 수행
        '''
        return await self._node.call_mobile_move_to_location(location_id, target_pose, global_path, navigation_mode)
    
    async def arm_move_to_pose(self, pose_type):
        # Arm 자세 변경 명령
        return await self._node.call_arm_move_to_pose(pose_type)
    
    async def arm_check_product(self, bbox_number):
        # Arm 상품 체크 명령
        return await self._node.call_arm_check_product(bbox_number)
    
    async def arm_pick_product(self, product_id, target_position):
        # Arm 상품 픽업 명령
        return await self._node.call_arm_pick_product(product_id, target_position)
    
    async def arm_place_product(self, product_id):
        # Arm 상품 놓기 명령
        return await self._node.call_arm_place_product(product_id)
    
    async def vision_detect_products(self, product_ids):
        # Vision 상품 인식 명령
        return await self._node.call_vision_detect_products(product_ids)
    
    async def vision_set_mode(self, mode):
        # Vision 모드 설정 명령
        return await self._node.call_vision_set_mode(mode)
    
    async def vision_track_staff(self, track):
        # Vision 직원 추종 제어 명령
        return await self._node.call_vision_track_staff(track)
    
    # Main Service에 보고하기 위한 Publisher 헬퍼 메소드들
    def publish_arrival_notice(self, location_id, section_id=0):
        # 목적지 도착 알림 발행
        self._node.publish_arrival_notice(location_id, section_id)
    
    def publish_product_detected(self, products):
        # 상품 인식 완료 알림 발행
        self._node.publish_product_detected(products)
    
    def publish_cart_handover_complete(self):
        # 장바구니 교체 완료 알림 발행
        self._node.publish_cart_handover_complete()
    
    def publish_product_selection_result(self, product_id, success, quantity, message=''):
        # 상품 담기 완료 보고 발행
        self._node.publish_product_selection_result(product_id, success, quantity, message)
    
    async def get_product_location(self, product_id):
        # Main Service에서 상품 위치 조회
        return await self._node.get_product_location(product_id)
