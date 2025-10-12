from .state import State


class WaitingLoadingState(State):
    """적재대기중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering WAITING_LOADING state')
        # TODO: 직원이 상품을 장바구니에 싣는 것을 대기
        
    def execute(self):
        # TODO: 적재 대기 로직
        # 음성 명령(e.g., "매대로 가자") 인식 시 상태 전환
        pass
    
    def on_exit(self):
        self._node.get_logger().info('Exiting WAITING_LOADING state')
