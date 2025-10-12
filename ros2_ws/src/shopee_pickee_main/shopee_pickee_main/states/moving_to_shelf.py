from .state import State
from shopee_interfaces.srv import MainGetLocationPose

class MovingToShelfState(State):
    """상품위치이동중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering MOVING_TO_SHELF state')
        self.target_location_id = getattr(self._node, 'target_location_id', 1)
        self.arrival_received = False
        self.pose_future = None
        self.is_moving = False

        # 1. 위치 좌표 요청
        self._node.get_logger().info(f'Requesting pose for location_id: {self.target_location_id}')
        request = MainGetLocationPose.Request()
        request.location_id = self.target_location_id
        self.pose_future = self._node.get_location_pose_client.call_async(request)

    def execute(self):
        # 2. 좌표를 받아 이동 명령을 내렸는지 확인
        if self.pose_future is not None:
            if self.pose_future.done():
                response = self.pose_future.result()
                if response and response.success:
                    target_pose = response.pose
                    self._node.get_logger().info(f'Received pose ({target_pose.x}, {target_pose.y}), commanding robot to move.')
                    
                    # 3. Mobile에 이동 명령 전달
                    self._node.call_mobile_move_to_location(self.target_location_id, target_pose)
                    self.is_moving = True
                else:
                    self._node.get_logger().error(f'Failed to get pose for location_id: {self.target_location_id}')
                    # TODO: 실패 상태로 전환하는 로직 필요
                
                self.pose_future = None # Future 처리 완료

        # 4. 이동 시작 후, 도착했는지 확인
        if self.is_moving:
            if hasattr(self._node, 'arrival_received') and self._node.arrival_received:
                self._node.get_logger().info('Arrived at shelf, starting product detection')
                self._node.arrival_received = False  # 플래그 리셋
                
                from .detecting_product import DetectingProductState
                new_state = DetectingProductState(self._node)
                self._node.state_machine.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('Exiting MOVING_TO_SHELF state')
