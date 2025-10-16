import rclpy
from rclpy.node import Node
from pickee_mobile.state_machine import StateMachine
from pickee_mobile.states import IdleState, MovingState
from pickee_mobile.localization_component import LocalizationComponent
from pickee_mobile.path_planning_component import PathPlanningComponent
from pickee_mobile.motion_control_component import MotionControlComponent

class MobileController(Node):
    '''
    Pickee Mobile의 메인 컨트롤러 노드.
    Pickee Main Controller의 지시에 따라 로봇의 이동을 제어하고 상태를 보고합니다.
    '''

    def __init__(self):
        super().__init__('pickee_mobile_controller')
        self.get_logger().info('Pickee Mobile Controller 노드가 시작되었습니다.')

        # 파라미터 선언
        self.declare_parameter('arrival_position_tolerance', 0.05)

        # 컴포넌트 초기화
        self.localization_component = LocalizationComponent(self)
        self.path_planning_component = PathPlanningComponent(self)
        self.motion_control_component = MotionControlComponent(self)

        # 상태 기계 초기화
        self.state_machine = StateMachine(IdleState(self), self)

        # 100ms 주기 상태 업데이트 타이머 생성
        self.timer = self.create_timer(0.1, self.update_state)

    def update_state(self):
        '''
        상태 기계의 현재 상태를 업데이트하고, Localization, Path Planning, Motion Control 컴포넌트를 업데이트합니다.
        예외 발생 시 노드 종료 없이 로깅 후 계속 실행합니다.
        '''
        try:
            # 상태 기계 실행
            self.state_machine.execute_current_state()

            # 현재 상태 이름 전달 (각 상태 클래스에 name 속성 사용)
            state_name = getattr(self.state_machine.current_state, 'name', 'UNKNOWN')
            self.localization_component.update_pose(state_name)

            # Path Planning 컴포넌트 업데이트
            current_pose = self.localization_component.get_current_pose()
            self.path_planning_component.plan_local_path(current_pose)

            # Motion Control 컴포넌트 업데이트
            self.motion_control_component.control_robot()

        except Exception as e:
            self.get_logger().error(f'update_state 실행 중 오류 발생: {e}')

def main(args=None):
    rclpy.init(args=args)
    try:
        mobile_controller = MobileController()
        rclpy.spin(mobile_controller)
    except KeyboardInterrupt:
        pass
    finally:
        mobile_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
