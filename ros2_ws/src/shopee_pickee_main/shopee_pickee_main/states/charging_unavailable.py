from .state import State
from .charging_available import ChargingAvailableState


class ChargingUnavailableState(State):
    """충전중 (작업불가) 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering CHARGING_UNAVAILABLE state')
        self._node.get_logger().info('Battery too low, charging...')
        
    def execute(self):
        # 배터리 상태 확인
        battery_level = self._node.current_battery_level
        battery_threshold = self._node.get_parameter('battery_threshold_available').get_parameter_value().double_value
        
        if battery_level >= battery_threshold:
            self._node.get_logger().info(f'Battery charged to {battery_level}%, transitioning to CHARGING_AVAILABLE')
            
            # CHARGING_AVAILABLE 상태로 전환
            new_state = ChargingAvailableState(self._node)
            self._node.state_machine.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('Exiting CHARGING_UNAVAILABLE state')
