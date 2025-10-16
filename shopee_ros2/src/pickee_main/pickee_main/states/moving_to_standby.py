from .state import State


class MovingToStandbyState(State):
    # 대기장소이동중 상태
    
    def on_enter(self):
        self._node.get_logger().info('MOVING_TO_STANDBY 상태 진입')
        self.standby_location_id = getattr(self._node, 'base_location_id', 0)  # 기본 대기 위치
        self.is_moving = False
        self.move_started = False
        
    def execute(self):
        # 대기 장소 좌표 조회 및 이동
        if not self.move_started:
            self._node.get_logger().info(f'대기 장소 ID {self.standby_location_id}로 복귀합니다.')
            
            import threading
            import asyncio
            
            def move_to_standby():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # 1. 대기 장소 좌표 조회
                    standby_pose = loop.run_until_complete(
                        self._node.call_get_location_pose(self.standby_location_id)
                    )
                    
                    if standby_pose:
                        self._node.get_logger().info(f'대기 장소 좌표 조회 성공: ({standby_pose.x}, {standby_pose.y})')
                        
                        # 2. Mobile에 이동 명령 전달
                        success = loop.run_until_complete(
                            self._node.call_mobile_move_to_location(self.standby_location_id, standby_pose)
                        )
                        
                        if success:
                            self.is_moving = True
                            self._node.get_logger().info('대기 장소로 이동을 시작했습니다.')
                            # 이동 상태 발행
                            self._node.publish_moving_status(self.standby_location_id)
                        else:
                            self._node.get_logger().error('대기 장소 이동 명령 전달 실패')
                    else:
                        self._node.get_logger().error('대기 장소 좌표 조회 실패')
                    
                    loop.close()
                except Exception as e:
                    self._node.get_logger().error(f'대기 장소 이동 실패: {str(e)}')
            
            threading.Thread(target=move_to_standby).start()
            self.move_started = True
        
        # 도착 확인
        if (hasattr(self._node, 'arrival_received') and self._node.arrival_received and 
            hasattr(self._node, 'arrived_location_id') and self._node.arrived_location_id == self.standby_location_id):
            
            self._node.arrival_received = False  # 플래그 리셋
            self._node.get_logger().info('대기 장소에 도착했습니다. CHARGING_AVAILABLE 상태로 전환합니다.')
            
            # 도착 알림 발행
            self._node.publish_arrival_notice(self.standby_location_id)
            
            from .charging_available import ChargingAvailableState
            new_state = ChargingAvailableState(self._node)
            self.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('MOVING_TO_STANDBY 상태 탈출')