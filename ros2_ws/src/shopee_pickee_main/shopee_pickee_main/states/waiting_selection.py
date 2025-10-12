from .state import State


class WaitingSelectionState(State):
    """상품선택대기중 상태"""
    
    def on_enter(self):
        self._node.get_logger().info('Entering WAITING_SELECTION state')
        self._node.get_logger().info('Waiting for user product selection...')
        
    def execute(self):
        # /pickee/product/process_selection 서비스 수신 확인
        if hasattr(self._node, 'selection_request') and self._node.selection_request:
            selection_request = self._node.selection_request
            self._node.selection_request = None  # 플래그 리셋
            
            self._node.get_logger().info(f'Product selection received: product_id={selection_request.product_id}, bbox_number={selection_request.bbox_number}')
            
            # 선택된 상품 정보 저장
            self._node.selected_product_id = selection_request.product_id
            self._node.selected_bbox_number = selection_request.bbox_number
            
            # PICKING_PRODUCT 상태로 전환
            from .picking_product import PickingProductState
            new_state = PickingProductState(self._node)
            self._node.state_machine.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('Exiting WAITING_SELECTION state')
