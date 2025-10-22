from .state import State


class WaitingUnloadingState(State):
    # 하차대기중 상태
    
    def on_enter(self):
        self._node.get_logger().info('WAITING_UNLOADING 상태 진입')
        self.unloading_completed = False
        # 직원이 상품을 매대에 진열하는 것을 대기
        self._node.get_logger().info('직원이 상품을 매대에 진열하는 것을 기다리고 있습니다.')
        
    def execute(self):
        # 하차 완료를 위한 대기 로직
        # Vision을 통해 장바구니 비움 확인이나 직원의 신호를 대기
        
        # 임시로 시간 기반 완료 처리 (실제로는 Vision이나 사용자 입력 필요)
        if not hasattr(self, 'unloading_start_time'):
            import time
            self.unloading_start_time = time.time()
        
        # 15초 후 자동으로 하차 완료로 간주 (테스트용)
        import time
        if time.time() - self.unloading_start_time > 15.0:
            self._node.get_logger().info('상품 하차가 완료되었습니다.')
            
            # 하차 완료 후 창고로 복귀하거나 다음 작업 수행
            # 여기서는 직원을 계속 추종하도록 설정
            from .following_staff import FollowingStaffState
            return FollowingStaffState(self._node)
    
    def on_exit(self):
        self._node.get_logger().info('WAITING_UNLOADING 상태 탈출')