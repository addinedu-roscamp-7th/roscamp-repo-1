from .state import State


class MovingToWarehouseState(State):
    # 창고이동중 상태
    
    def on_enter(self):
        self._node.get_logger().info('MOVING_TO_WAREHOUSE 상태 진입')
        self.warehouse_id = getattr(self._node, 'target_warehouse_id', 1)
        self.is_moving = False
        self.move_started = False
        
    def execute(self):
        # 창고 좌표 조회 및 이동
        if not self.move_started:
            self._node.get_logger().info(f'창고 ID {self.warehouse_id}의 좌표를 조회합니다.')
            
            import threading
            import asyncio
            
            def move_to_warehouse():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # 1. 창고 좌표 조회
                    warehouse_pose = loop.run_until_complete(
                        self._node.call_get_warehouse_pose(self.warehouse_id)
                    )
                    
                    if warehouse_pose:
                        self._node.get_logger().info(f'창고 좌표 조회 성공: ({warehouse_pose.x}, {warehouse_pose.y})')
                        
                        # 2. Mobile에 이동 명령 전달
                        success = loop.run_until_complete(
                            self._node.call_mobile_move_to_location(self.warehouse_id, warehouse_pose)
                        )
                        
                        if success:
                            self.is_moving = True
                            self._node.get_logger().info('창고로 이동을 시작했습니다.')
                        else:
                            self._node.get_logger().error('창고 이동 명령 전달 실패')
                    else:
                        self._node.get_logger().error('창고 좌표 조회 실패')
                    
                    loop.close()
                except Exception as e:
                    self._node.get_logger().error(f'창고 이동 실패: {str(e)}')
            
            threading.Thread(target=move_to_warehouse).start()
            self.move_started = True
        
        # 도착 확인 (mobile_arrival_callback에서 처리)
        if (hasattr(self._node, 'arrival_received') and self._node.arrival_received and 
            hasattr(self._node, 'arrived_location_id') and self._node.arrived_location_id == self.warehouse_id):
            
            self._node.arrival_received = False  # 플래그 리셋
            self._node.get_logger().info('창고에 도착했습니다. WAITING_LOADING 상태로 전환합니다.')
            
            from .waiting_loading import WaitingLoadingState
            return WaitingLoadingState(self._node)
    
    def on_exit(self):
        self._node.get_logger().info('MOVING_TO_WAREHOUSE 상태 탈출')