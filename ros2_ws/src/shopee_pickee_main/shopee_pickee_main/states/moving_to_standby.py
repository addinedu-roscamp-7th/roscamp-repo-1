from .state import State


class MovingToStandbyState(State):
    """대기장소이동중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering MOVING_TO_STANDBY state')
        # TODO: 지정된 대기 장소로 복귀
        
    def execute(self):
        # TODO: 대기장소 이동 로직
        # /pickee/mobile/arrival 토픽 수신 시 CHARGING_UNAVAILABLE 상태로 전환
        pass
    
    def on_exit(self):
        self._node.get_logger().info('Exiting MOVING_TO_STANDBY state')
