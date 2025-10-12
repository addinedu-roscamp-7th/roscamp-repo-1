from .state import State


class ChargingAvailableState(State):
    """충전중 (작업가능) 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering CHARGING_AVAILABLE state')
        self._node.get_logger().info('Ready for task assignment')
        
    def execute(self):
        # /start_task 서비스 수신은 main_controller의 콜백에서 처리됨
        # 배터리 레벨이 임계값 이하로 떨어지면 CHARGING_UNAVAILABLE로 전환
        battery_level = self._node.current_battery_level
        battery_threshold = self._node.get_parameter('battery_threshold_unavailable').get_parameter_value().double_value
        
        if battery_level <= battery_threshold:
            self._node.get_logger().warn(f'Battery low ({battery_level}%), returning to charging')
            
            from .charging_unavailable import ChargingUnavailableState
            new_state = ChargingUnavailableState(self._node)
            self._node.state_machine.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('Exiting CHARGING_AVAILABLE state')
