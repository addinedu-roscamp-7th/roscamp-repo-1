from .state import State


class RegisteringStaffState(State):
    # 직원등록중 상태
    
    def on_enter(self):
        self._node.get_logger().info('REGISTERING_STAFF 상태 진입')
        # TODO: 재고 보충 직원 등록
        
    def execute(self):
        # TODO: 직원 등록 로직
        # /pickee/vision/register_staff_result (success) 토픽 수신 시 FOLLOWING_STAFF 상태로 전환
        pass
    
    def on_exit(self):
        self._node.get_logger().info('REGISTERING_STAFF 상태 탈출')
