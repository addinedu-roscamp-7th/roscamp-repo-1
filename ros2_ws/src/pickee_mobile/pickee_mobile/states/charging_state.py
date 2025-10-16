from pickee_mobile.state_machine import State
import rclpy

class ChargingState(State):
    '''
    Pickee Mobile의 충전 상태.
    배터리 충전을 수행합니다.
    '''

    def __init__(self, node: rclpy.node.Node):
        super().__init__(node)

    def on_enter(self):
        self.node.get_logger().info('CHARGING 상태 진입: 배터리 충전 시작.')

    def execute(self):
        # 충전 로직 (배터리 잔량 모니터링 등)
        pass

    def on_exit(self):
        self.node.get_logger().info('CHARGING 상태 이탈.')
