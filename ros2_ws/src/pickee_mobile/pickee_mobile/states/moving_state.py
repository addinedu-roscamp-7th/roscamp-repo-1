from pickee_mobile.state_machine import State
import rclpy

class MovingState(State):
    '''
    Pickee Mobile의 이동 상태.
    지정된 목적지로 이동합니다.
    '''

    def __init__(self, node: rclpy.node.Node):
        super().__init__(node)

    def on_enter(self):
        self.node.get_logger().info('MOVING 상태 진입: 목적지로 이동 시작.')

    def execute(self):
        # 이동 로직 (경로 계획, 추종, 장애물 회피 등)
        pass

    def on_exit(self):
        self.node.get_logger().info('MOVING 상태 이탈.')
