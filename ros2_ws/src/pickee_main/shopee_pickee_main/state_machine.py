class StateMachine:
    # 상태를 관리하고 상태 간의 전환을 처리하는 클래스
    def __init__(self, initial_state):
        self._current_state = initial_state
        if self._current_state:
            self._current_state.on_enter()

    def transition_to(self, new_state):
        # 새로운 상태로 전환합니다.
        if self._current_state:
            self._current_state.on_exit()
        
        self._current_state = new_state
        self._current_state.on_enter()

    def execute(self):
        # 현재 상태의 execute 메소드를 호출합니다.
        if self._current_state:
            self._current_state.execute()
    
    def get_current_state_name(self):
        # 현재 상태의 클래스 이름을 반환합니다.
        if self._current_state:
            return self._current_state.__class__.__name__
        return 'None'
