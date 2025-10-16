import unittest
from unittest.mock import Mock, MagicMock
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pickee_main.state_machine import StateMachine
from pickee_main.states.state import State

class MockState(State):
    """테스트용 Mock 상태 클래스"""
    def __init__(self, name="MockState", node=None):
        super().__init__(node or Mock())
        self.name = name
        self.on_enter_called = False
        self.execute_called = False
        self.on_exit_called = False
    
    def on_enter(self):
        self.on_enter_called = True
    
    def execute(self):
        self.execute_called = True
    
    def on_exit(self):
        self.on_exit_called = True
    
    def get_name(self):
        return self.name


class TestStateMachine(unittest.TestCase):
    """StateMachine 클래스에 대한 단위 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.initial_state = MockState("INITIAL")
        self.state_machine = StateMachine(self.initial_state)
    
    def test_initial_state_setup(self):
        """초기 상태가 올바르게 설정되는지 테스트"""
        # 초기 상태가 올바르게 설정되었는지 확인
        self.assertEqual(self.state_machine.get_current_state_name(), "MockState")
        
        # 초기 상태의 on_enter가 호출되었는지 확인
        self.assertTrue(self.initial_state.on_enter_called)
    
    def test_state_transition(self):
        """상태 전환이 올바르게 동작하는지 테스트"""
        new_state = MockState("NEW_STATE")
        
        # 상태 전환 실행
        self.state_machine.transition_to(new_state)
        
        # 이전 상태의 on_exit이 호출되었는지 확인
        self.assertTrue(self.initial_state.on_exit_called)
        
        # 새 상태의 on_enter가 호출되었는지 확인
        self.assertTrue(new_state.on_enter_called)
        
        # 현재 상태가 변경되었는지 확인
        self.assertEqual(self.state_machine.get_current_state_name(), "MockState")
    
    def test_execute(self):
        """상태 기계의 execute가 현재 상태의 execute를 호출하는지 테스트"""
        # execute 호출 전에는 실행되지 않았는지 확인
        self.assertFalse(self.initial_state.execute_called)
        
        # execute 실행
        self.state_machine.execute()
        
        # 현재 상태의 execute가 호출되었는지 확인
        self.assertTrue(self.initial_state.execute_called)
    
    def test_state_machine_with_none_state(self):
        """None 상태로 초기화했을 때의 동작 테스트"""
        state_machine = StateMachine(None)
        
        # None 상태일 때 안전하게 동작하는지 확인
        self.assertEqual(state_machine.get_current_state_name(), "None")
        
        # execute가 예외를 발생시키지 않는지 확인
        try:
            state_machine.execute()
        except Exception as e:
            self.fail(f"execute() raised {e} unexpectedly!")


if __name__ == '__main__':
    # 테스트 실행
    unittest.main()