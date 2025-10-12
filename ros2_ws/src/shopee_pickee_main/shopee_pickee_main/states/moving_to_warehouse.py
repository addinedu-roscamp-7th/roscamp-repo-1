from .state import State


class MovingToWarehouseState(State):
    """창고이동중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering MOVING_TO_WAREHOUSE state')
        # TODO: 재고 보충을 위해 창고로 이동
        
    def execute(self):
        # TODO: 창고 이동 로직
        # /pickee/mobile/arrival 토픽 수신 시 WAITING_LOADING 상태로 전환
        pass
    
    def on_exit(self):
        self._node.get_logger().info('Exiting MOVING_TO_WAREHOUSE state')
