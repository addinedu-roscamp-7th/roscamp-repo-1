from .state import State


class MovingToShelfState(State):
    # 상품위치이동중 상태
    
    def on_enter(self):
        self._node.get_logger().info('MOVING_TO_SHELF 상태 진입')
        self.target_location_id = getattr(self._node, 'target_location_id', 1)
        self.is_moving = False
        self.move_started = False

    def execute(self):
        # 선반 좌표 조회 및 이동
        if not self.move_started:
            self._node.get_logger().info(f'선반 ID {self.target_location_id}로 이동을 시작합니다.')
            
            import threading
            import asyncio
            
            def move_to_shelf():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # 1. 선반 좌표 조회
                    shelf_pose = loop.run_until_complete(
                        self._node.call_get_location_pose(self.target_location_id)
                    )
                    
                    if shelf_pose:
                        self._node.get_logger().info(f'선반 좌표 조회 성공: ({shelf_pose.x}, {shelf_pose.y})')
                        
                        # 2. Mobile에 이동 명령 전달
                        success = loop.run_until_complete(
                            self._node.call_mobile_move_to_location(self.target_location_id, shelf_pose)
                        )
                        
                        if success:
                            self.is_moving = True
                            self._node.get_logger().info('선반으로 이동을 시작했습니다.')
                            # 이동 상태 발행
                            self._node.publish_moving_status(self.target_location_id)
                        else:
                            self._node.get_logger().error('선반 이동 명령 전달 실패')
                    else:
                        self._node.get_logger().error('선반 좌표 조회 실패')
                    
                    loop.close()
                except Exception as e:
                    self._node.get_logger().error(f'선반 이동 실패: {str(e)}')
            
            threading.Thread(target=move_to_shelf).start()
            self.move_started = True
        
        # 도착 확인
        if (hasattr(self._node, 'arrival_received') and self._node.arrival_received and 
            hasattr(self._node, 'arrived_location_id') and self._node.arrived_location_id == self.target_location_id):
            
            self._node.arrival_received = False  # 플래그 리셋
            self._node.get_logger().info('선반에 도착했습니다. 상품 인식을 시작합니다.')
            
            # 선반 도착 알림 발행
            self._node.publish_arrival_notice(self.target_location_id, section_id=7)
            
            from .detecting_product import DetectingProductState
            return DetectingProductState(self._node)
    
    def on_exit(self):
        self._node.get_logger().info('MOVING_TO_SHELF 상태 탈출')