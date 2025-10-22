from .state import State
from .charging_unavailable import ChargingUnavailableState


class InitializingState(State):
    # 초기화중 상태
    
    def on_enter(self):
        self._node.get_logger().info('INITIALIZING 상태 진입')
        self.initialization_complete = False
        self.initialization_timer = 0
        
    def execute(self):
        # 초기화 시뮬레이션 (3초 후 완료)
        self.initialization_timer += 1
        
        if self.initialization_timer >= 30:  # 3초 (10Hz 타이머 기준)
            self._node.get_logger().info('초기화 완료')
            self.initialization_complete = True
            
            # CHARGING_UNAVAILABLE 상태로 전환
            return ChargingUnavailableState(self._node)
    
    def on_exit(self):
        self._node.get_logger().info('INITIALIZING 상태 탈출')
