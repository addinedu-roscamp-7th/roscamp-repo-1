from .state import State


class DetectingProductState(State):
    """상품인식중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering DETECTING_PRODUCT state')
        self.detection_started = False
        self.detection_completed = False
        
        # Arm을 shelf_view 자세로 변경
        self._node.get_logger().info('Moving arm to shelf view position')
        # TODO: await self.arm_move_to_pose("shelf_view")
        
        # Vision을 detect_products 모드로 설정
        self._node.get_logger().info('Setting vision mode to detect_products')
        # TODO: await self.vision_set_mode("detect_products")
        
    def execute(self):
        if not self.detection_started:
            # 상품 인식 시작
            target_product_ids = getattr(self._node, 'target_product_ids', [1, 2, 3])
            self._node.get_logger().info(f'Starting product detection for: {target_product_ids}')
            # TODO: await self.vision_detect_products(target_product_ids)
            self.detection_started = True
        
        # /pickee/vision/detection_result 토픽 수신 확인
        if hasattr(self._node, 'detection_result') and self._node.detection_result:
            detection_result = self._node.detection_result
            self._node.detection_result = None  # 플래그 리셋
            
            # 인식 결과를 Main Service에 보고
            self.publish_product_detected(detection_result.products)
            
            # 자동/수동 선택 모드에 따라 상태 전환
            selection_mode = getattr(self._node, 'selection_mode', 'manual')
            
            if selection_mode == 'auto' and detection_result.products:
                # 자동 선택: 가장 신뢰도 높은 상품 선택
                best_product = max(detection_result.products, key=lambda p: p.confidence)
                self._node.selected_product = best_product
                
                from .picking_product import PickingProductState
                new_state = PickingProductState(self._node)
                self._node.state_machine.transition_to(new_state)
            else:
                # 수동 선택 대기
                from .waiting_selection import WaitingSelectionState
                new_state = WaitingSelectionState(self._node)
                self._node.state_machine.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('Exiting DETECTING_PRODUCT state')
