from .state import State


class MovingToPackingState(State):
    """포장대이동중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering MOVING_TO_PACKING state')
        # TODO: 포장대로 이동
        
    def execute(self):
        # TODO: 포장대 이동 로직
        # /pickee/mobile/arrival 토픽 수신 시 WAITING_HANDOVER 상태로 전환
        pass
    
    def on_exit(self):
        self._node.get_logger().info('Exiting MOVING_TO_PACKING state')
