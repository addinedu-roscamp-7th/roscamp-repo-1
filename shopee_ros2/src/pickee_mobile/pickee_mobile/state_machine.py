from abc import ABC, abstractmethod
import rclpy

class State(ABC):
    '''
    상태 기계의 개별 상태를 나타내는 추상 기본 클래스.
    모든 구체적인 상태 클래스는 이 클래스를 상속받아야 합니다.
    '''

    def __init__(self, node: rclpy.node.Node):
        self.node = node

    @abstractmethod
    def on_enter(self):
        '''
        상태 진입 시 실행되는 로직.
        '''
        pass

    @abstractmethod
    def execute(self):
        '''
        상태가 활성화되어 있는 동안 주기적으로 실행되는 로직.
        '''
        pass

    @abstractmethod
    def on_exit(self):
        '''
        상태 이탈 시 실행되는 로직.
        '''
        pass

class StateMachine:
    '''
    로봇의 상태를 관리하고 상태 간의 전환을 처리하는 상태 기계 클래스.
    '''

    def __init__(self, initial_state: State, node: rclpy.node.Node):
        self.node = node
        self.current_state = initial_state
        self.current_state.on_enter()
        self.node.get_logger().info(f'상태 기계 초기화: {type(self.current_state).__name__}')

    def transition_to(self, new_state: State):
        '''
        새로운 상태로 전환합니다.
        현재 상태의 on_exit를 호출하고, 새 상태의 on_enter를 호출합니다.
        '''
        self.node.get_logger().info(f'상태 전환: {type(self.current_state).__name__} -> {type(new_state).__name__}')
        self.current_state.on_exit()
        self.current_state = new_state
        self.current_state.on_enter()

    def execute_current_state(self):
        '''
        현재 상태의 execute 로직을 실행합니다.
        '''
        self.current_state.execute()
