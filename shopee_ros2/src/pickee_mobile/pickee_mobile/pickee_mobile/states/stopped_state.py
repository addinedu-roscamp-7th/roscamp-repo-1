from pickee_mobile.state_machine import State
import rclpy

class StoppedState(State):
    '''
    Pickee Mobile의 정지 상태.
    이동 중단 (장애물, 명령 등) 시 진입합니다.
    '''

    def __init__(self, node: rclpy.node.Node):
        super().__init__(node)

    def on_enter(self):
        self.node.get_logger().info('STOPPED 상태 진입: 로봇 정지.')

    def execute(self):
        # 정지 상태에서 주기적으로 수행할 작업 (예: 주변 장애물 감시)
        pass

    def on_exit(self):
        self.node.get_logger().info('STOPPED 상태 이탈.')
