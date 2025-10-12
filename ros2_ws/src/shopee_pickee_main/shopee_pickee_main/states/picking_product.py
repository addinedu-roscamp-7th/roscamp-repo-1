from .state import State


class PickingProductState(State):
    """상품담기중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering PICKING_PRODUCT state')
        self.pick_started = False
        self.pick_completed = False
        self.place_started = False
        self.place_completed = False
        
        # 선택된 상품 정보
        self.product_id = getattr(self._node, 'selected_product_id', 1)
        self.target_position = getattr(self._node, 'selected_target_position', None)
        
    def execute(self):
        if not self.pick_started:
            # 상품 픽업 시작
            self._node.get_logger().info(f'Starting to pick product: {self.product_id}')
            # TODO: await self.arm_pick_product(self.product_id, self.target_position)
            self.pick_started = True
        
        elif not self.pick_completed and hasattr(self._node, 'arm_pick_completed') and self._node.arm_pick_completed:
            # 픽업 완료, 장바구니에 놓기 시작
            self._node.arm_pick_completed = False  # 플래그 리셋
            self._node.get_logger().info(f'Pick completed, placing product in cart: {self.product_id}')
            # TODO: await self.arm_place_product(self.product_id)
            self.pick_completed = True
            self.place_started = True
        
        elif self.place_started and hasattr(self._node, 'arm_place_completed') and self._node.arm_place_completed:
            # 장바구니에 놓기 완료
            self._node.arm_place_completed = False  # 플래그 리셋
            self._node.get_logger().info(f'Product placed in cart successfully: {self.product_id}')
            
            # Main Service에 결과 보고
            self.publish_product_selection_result(self.product_id, True, 1, "Product picked successfully")
            
            # 다음 상품이 있는지 확인
            if hasattr(self._node, 'remaining_products') and self._node.remaining_products:
                # 다음 매대로 이동
                from .moving_to_shelf import MovingToShelfState
                new_state = MovingToShelfState(self._node)
                self._node.state_machine.transition_to(new_state)
            else:
                # 모든 상품을 담았으면 쇼핑 종료 대기
                from .charging_available import ChargingAvailableState
                new_state = ChargingAvailableState(self._node)
                self._node.state_machine.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('Exiting PICKING_PRODUCT state')
