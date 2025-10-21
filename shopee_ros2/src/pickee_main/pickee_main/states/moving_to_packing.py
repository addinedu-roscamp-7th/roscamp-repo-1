from .state import State


class MovingToPackingState(State):
    # 포장대이동중 상태
    
    def on_enter(self):
        self._node.get_logger().info('MOVING_TO_PACKING 상태 진입')
        self.packing_location_id = getattr(self._node, 'target_packaging_location_id', 100)  # 기본 포장대 ID
        self.is_moving = False
        self.move_started = False
        
    def execute(self):
        # 포장대 좌표 조회 및 이동
        if not self.move_started:
            self._node.get_logger().info(f'포장대 ID {self.packing_location_id}로 이동을 시작합니다.')
            
            import threading
            import asyncio
            
            def move_to_packing():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # 1. 포장대 좌표 조회
                    packing_pose = loop.run_until_complete(
                        self._node.call_get_location_pose(self.packing_location_id)
                    )
                    
                    if packing_pose:
                        self._node.get_logger().info(f'포장대 좌표 조회 성공: ({packing_pose.x}, {packing_pose.y})')
                        
                        # 2. Mobile에 이동 명령 전달
                        success = loop.run_until_complete(
                            self._node.call_mobile_move_to_location(self.packing_location_id, packing_pose)
                        )
                        
                        if success:
                            self.is_moving = True
                            self._node.get_logger().info('포장대로 이동을 시작했습니다.')
                            # 이동 상태 발행
                            self._node.publish_moving_status(self.packing_location_id)
                        else:
                            self._node.get_logger().error('포장대 이동 명령 전달 실패')
                    else:
                        self._node.get_logger().error('포장대 좌표 조회 실패')
                    
                    loop.close()
                except Exception as e:
                    self._node.get_logger().error(f'포장대 이동 실패: {str(e)}')
            
            threading.Thread(target=move_to_packing).start()
            self.move_started = True
        
        # 도착 확인
        if (hasattr(self._node, 'arrival_received') and self._node.arrival_received and 
            hasattr(self._node, 'arrived_location_id') and self._node.arrived_location_id == self.packing_location_id):
            
            self._node.arrival_received = False  # 플래그 리셋
            self._node.get_logger().info('포장대에 도착했습니다. WAITING_HANDOVER 상태로 전환합니다.')
            
            # 포장대 도착 알림 발행
            self._node.publish_arrival_notice(self.packing_location_id)
            
            from .waiting_handover import WaitingHandoverState
            return WaitingHandoverState(self._node)
    
    def on_exit(self):
        self._node.get_logger().info('MOVING_TO_PACKING 상태 탈출')