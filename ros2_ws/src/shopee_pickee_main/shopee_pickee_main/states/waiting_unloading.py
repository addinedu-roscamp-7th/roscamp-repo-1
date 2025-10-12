from .state import State


class WaitingUnloadingState(State):
    """하차대기중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering WAITING_UNLOADING state')
        # TODO: 직원이 상품을 매대에 진열하는 것을 대기
        
    def execute(self):
        # TODO: 하차 대기 로직
        # 음성 명령(e.g., "다 끝났어") 인식 시 상태 전환
        pass
    
    def on_exit(self):
        self._node.get_logger().info('Exiting WAITING_UNLOADING state')
