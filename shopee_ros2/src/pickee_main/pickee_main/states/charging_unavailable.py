from .state import State
from .charging_available import ChargingAvailableState


class ChargingUnavailableState(State):
    # 충전중 (작업불가) 상태
    
    def on_enter(self):
        self._node.get_logger().info('CHARGING_UNAVAILABLE 상태 진입')
        self._node.get_logger().info('배터리 부족, 충전을 시작합니다.')
        
    def execute(self):
        # 배터리 상태 확인
        battery_level = self._node.current_battery_level
        battery_threshold = self._node.get_parameter('battery_threshold_available').get_parameter_value().double_value
        
        if battery_level >= battery_threshold:
            self._node.get_logger().info(f'배터리 충전 완료 ({battery_level}%), CHARGING_AVAILABLE 상태로 전환합니다.')
            
            # CHARGING_AVAILABLE 상태로 전환
            return ChargingAvailableState(self._node)
    
    def on_exit(self):
        self._node.get_logger().info('CHARGING_UNAVAILABLE 상태 탈출')
