import rclpy
from rclpy.node import Node
from pickee_mobile.state_machine import StateMachine
from pickee_mobile.states import IdleState

class MobileController(Node):
    '''
    Pickee Mobile의 메인 컨트롤러 노드.
    Pickee Main Controller의 지시에 따라 로봇의 이동을 제어하고 상태를 보고합니다.
    '''

    def __init__(self):
        super().__init__('pickee_mobile_controller')
        self.get_logger().info('Pickee Mobile Controller 노드가 시작되었습니다.')

        # 상태 기계 초기화
        self.state_machine = StateMachine(IdleState(self), self)
        self.timer = self.create_timer(0.1, self.update_state) # 100ms 주기로 상태 업데이트

    def update_state(self):
        '''
        상태 기계의 현재 상태를 업데이트합니다.
        '''
        self.state_machine.execute_current_state()

def main(args=None):
    rclpy.init(args=args)
    mobile_controller = MobileController()
    rclpy.spin(mobile_controller)
    mobile_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
