from pickee_mobile.state_machine import State
import rclpy

class IdleState(State):
    '''
    Pickee Mobile의 대기 상태.
    Pickee Main Controller의 명령을 기다립니다.
    '''

    def __init__(self, node: rclpy.node.Node):
        super().__init__(node)

    def on_enter(self):
        self.node.get_logger().info('IDLE 상태 진입: 명령 대기 중...')

    def execute(self):
        # 대기 상태에서 주기적으로 수행할 작업 (예: 배터리 상태 확인 등)
        pass

    def on_exit(self):
        self.node.get_logger().info('IDLE 상태 이탈.')
