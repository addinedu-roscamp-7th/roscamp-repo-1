from .state import State
import threading
import asyncio


class RegisteringStaffState(State):
    # 직원등록중 상태
    
    def on_enter(self):
        self._node.get_logger().info('REGISTERING_STAFF 상태 진입')
        self.registration_started = False
        # 직원 등록 플래그 초기화
        self._node.staff_registration_completed = False
        self._node.staff_registration_failed = False
        
    def execute(self):
        # 직원 등록 시작
        if not self.registration_started:
            self._node.get_logger().info('직원 등록을 시작합니다.')
            
            # Vision에 직원 등록 명령 전달
            def register_staff():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        self._node.call_vision_register_staff()
                    )
                    loop.close()
                    if not success:
                        self._node.get_logger().error('직원 등록 명령 전달 실패')
                        self._node.staff_registration_failed = True
                except Exception as e:
                    self._node.get_logger().error(f'직원 등록 실패: {str(e)}')
                    self._node.staff_registration_failed = True
            
            threading.Thread(target=register_staff).start()
            self.registration_started = True
            
        # 직원 등록 완료 확인 (vision_staff_register_callback에서 처리됨)
        if hasattr(self._node, 'staff_registration_completed') and self._node.staff_registration_completed:
            self._node.get_logger().info('직원 등록이 완료되었습니다. FOLLOWING_STAFF 상태로 전환합니다.')
            from .following_staff import FollowingStaffState
            new_state = FollowingStaffState(self._node)
            self.transition_to(new_state)
            
        elif hasattr(self._node, 'staff_registration_failed') and self._node.staff_registration_failed:
            self._node.get_logger().error('직원 등록에 실패했습니다. CHARGING_AVAILABLE 상태로 복귀합니다.')
            from .charging_available import ChargingAvailableState
            new_state = ChargingAvailableState(self._node)
            self.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('REGISTERING_STAFF 상태 탈출')