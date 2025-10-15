from .state import State


class WaitingHandoverState(State):
    # 장바구니전달대기중 상태
    
    def on_enter(self):
        self._node.get_logger().info('WAITING_HANDOVER 상태 진입')
        # TODO: Packee 로봇에게 장바구니 전달 대기
        
    def execute(self):
        # TODO: 장바구니 전달 대기 로직
        # /pickee/cart_handover_complete 토픽 수신 시 MOVING_TO_STANDBY 상태로 전환
        pass
    
    def on_exit(self):
        self._node.get_logger().info('WAITING_HANDOVER 상태 탈출')
