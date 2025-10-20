from .state import State


class WaitingHandoverState(State):
    # 장바구니전달대기중 상태
    
    def on_enter(self):
        self._node.get_logger().info('WAITING_HANDOVER 상태 진입')
        self.handover_completed = False
        # Packee 로봇에게 장바구니 전달 대기
        self._node.get_logger().info('Packee 로봇에게 장바구니를 전달하기를 기다리고 있습니다.')
        
    def execute(self):
        # 장바구니 전달 대기 로직
        # Packee 로봇과의 통신이나 물리적 전달 완료 신호 대기
        
        # 임시로 시간 기반 완료 처리 (실제로는 Packee와의 통신 필요)
        if not hasattr(self, 'handover_start_time'):
            import time
            self.handover_start_time = time.time()
        
        # 20초 후 자동으로 전달 완료로 간주 (테스트용)
        import time
        if time.time() - self.handover_start_time > 20.0:
            self._node.get_logger().info('장바구니 전달이 완료되었습니다.')
            
            # 장바구니 전달 완료 알림 발행
            self._node.publish_cart_handover_complete()
            
            # 전달 완료 후 충전 가능 상태로 복귀
            from .charging_available import ChargingAvailableState
            return ChargingAvailableState(self._node)
    
    def on_exit(self):
        self._node.get_logger().info('WAITING_HANDOVER 상태 탈출')