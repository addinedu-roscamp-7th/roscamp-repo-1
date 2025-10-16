import rclpy
from rclpy.node import Node
from pickee_mobile.state_machine import StateMachine
from pickee_mobile.states import IdleState, MovingState
from pickee_mobile.localization_component import LocalizationComponent
from pickee_mobile.path_planning_component import PathPlanningComponent
from pickee_mobile.motion_control_component import MotionControlComponent
from shopee_interfaces.msg import PickeeMobileArrival # 추가

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

        # Publisher 초기화
        self.arrival_publisher = self.create_publisher(
            PickeeMobileArrival,
            '/pickee/mobile/arrival',
            10
        )

        # 상태 기계 초기화
        self.state_machine = StateMachine(IdleState(self), self)

        # 100ms 주기 상태 업데이트 타이머 생성
        self.timer = self.create_timer(0.1, self.update_state)

        self.has_arrived = False # 도착 여부 플래그

    def update_state(self):
        '''
        상태 기계의 현재 상태를 업데이트하고, Localization, Path Planning, Motion Control 컴포넌트를 업데이트합니다.
        예외 발생 시 노드 종료 없이 로깅 후 계속 실행합니다.
        '''
        try:
            # 상태 기계 실행
            self.state_machine.execute_current_state()

            # 현재 상태 이름 전달 (각 상태 클래스에 name 속성 사용)
            state_name = type(self.state_machine.current_state).__name__.replace('State', '').upper()
            self.localization_component.update_pose(state_name)

            # Path Planning 컴포넌트 업데이트
            current_pose = self.localization_component.get_current_pose()
            move_status = self.path_planning_component.plan_local_path(current_pose) # plan_local_path가 move_status를 반환하도록 수정 필요

            # Motion Control 컴포넌트 업데이트
            self.motion_control_component.control_robot()

            # 도착 여부 확인 및 메시지 발행
            if move_status and move_status.is_arrived and not self.has_arrived:
                self.publish_arrival(move_status)
                self.has_arrived = True
                self.state_machine.transition_to(IdleState(self)) # 도착 시 IDLE 상태로 전환
            elif move_status and not move_status.is_arrived:
                self.has_arrived = False # 도착 상태 초기화

        except Exception as e:
            self.get_logger().error(f'update_state 실행 중 오류 발생: {e}')

    def publish_arrival(self, move_status):
        '''
        도착 메시지를 발행합니다.
        '''
        arrival_msg = PickeeMobileArrival()
        arrival_msg.header.stamp = self.get_clock().now().to_msg()
        arrival_msg.final_x = move_status.current_x
        arrival_msg.final_y = move_status.current_y
        arrival_msg.final_theta = move_status.current_theta
        arrival_msg.position_error = move_status.distance_to_target
        # arrival_msg.travel_time = ... # 이동 시간은 현재 계산하지 않음
        self.arrival_publisher.publish(arrival_msg)
        self.get_logger().info(f'도착 메시지 발행: final_x={arrival_msg.final_x:.2f}, error={arrival_msg.position_error:.2f}')

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
