from .state import State


class MovingToShelfState(State):
    """상품위치이동중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering MOVING_TO_SHELF state')
        self.target_location_id = getattr(self._node, 'target_location_id', 1)
        self.arrival_received = False
        
        # Mobile에 이동 명령 전달
        self._node.get_logger().info(f'Moving to shelf location: {self.target_location_id}')
        # TODO: 실제 Mobile 이동 명령 호출
        # await self.mobile_move_to_location(self.target_location_id, target_pose)
        
    def execute(self):
        # /pickee/mobile/arrival 토픽 수신은 main_controller의 콜백에서 처리됨
        # 도착 알림을 받으면 DETECTING_PRODUCT 상태로 전환
        if hasattr(self._node, 'arrival_received') and self._node.arrival_received:
            self._node.get_logger().info('Arrived at shelf, starting product detection')
            self._node.arrival_received = False  # 플래그 리셋
            
            from .detecting_product import DetectingProductState
            new_state = DetectingProductState(self._node)
            self._node.state_machine.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('Exiting MOVING_TO_SHELF state')
