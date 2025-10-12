from .state import State


class FollowingStaffState(State):
    """직원추종중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering FOLLOWING_STAFF state')
        # TODO: 직원을 따라 이동
        
    def execute(self):
        # TODO: 직원 추종 로직
        # 음성 명령(e.g., "창고로 가자") 인식 시 상태 전환
        pass
    
    def on_exit(self):
        self._node.get_logger().info('Exiting FOLLOWING_STAFF state')
