from .state import State
import threading
import asyncio


class FollowingStaffState(State):
    # 직원추종중 상태
    
    def on_enter(self):
        self._node.get_logger().info('FOLLOWING_STAFF 상태 진입')
        self.tracking_started = False
        # 직원 추종을 위한 Vision 트래킹 시작
        
    def execute(self):
        # 직원 트래킹 시작
        if not self.tracking_started:
            self._node.get_logger().info('직원 트래킹을 시작합니다.')
            
            # Vision에 직원 추종 명령 전달
            def start_tracking():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        self._node.call_vision_track_staff(True)  # 트래킹 시작
                    )
                    loop.close()
                    if success:
                        self._node.get_logger().info('직원 추종 모드가 활성화되었습니다.')
                    else:
                        self._node.get_logger().error('직원 추종 모드 활성화 실패')
                except Exception as e:
                    self._node.get_logger().error(f'직원 추종 시작 실패: {str(e)}')
            
            threading.Thread(target=start_tracking).start()
            self.tracking_started = True
        
        # 직원 위치 정보가 수신되면 자동으로 vision_staff_location_callback에서 처리됨
        # 음성 명령을 통한 상태 전환은 향후 LLM 서비스 연동 시 구현
        
        # 예시: 특정 조건에서 창고로 이동 (음성 명령 대신 임시 로직)
        # 실제로는 LLM 서비스에서 음성을 텍스트로 변환하고 명령을 분석해야 함
        pass
    
    def on_exit(self):
        self._node.get_logger().info('FOLLOWING_STAFF 상태 탈출')