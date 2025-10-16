from pickee_mobile.state_machine import State
import rclpy

class ErrorState(State):
    '''
    Pickee Mobile의 오류 상태.
    시스템 오류 발생 시 진입합니다.
    '''

    def __init__(self, node: rclpy.node.Node):
        super().__init__(node)

    def on_enter(self):
        self.node.get_logger().error('ERROR 상태 진입: 시스템 오류 발생!')

    def execute(self):
        # 오류 처리 로직 (예: 오류 보고, 안전 모드 유지)
        pass

    def on_exit(self):
        self.node.get_logger().info('ERROR 상태 이탈: 오류 해결 또는 재시작.')
