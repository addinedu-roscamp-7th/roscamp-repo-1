from .state import State


class WaitingSelectionState(State):
    # 상품선택대기중 상태
    
    def on_enter(self):
        self._node.get_logger().info('WAITING_SELECTION 상태 진입')
        self._node.get_logger().info('상품 선택 대기 중...')

    def execute(self):
        # /pickee/product/process_selection 서비스 수신 확인
        if hasattr(self._node, 'selection_request') and self._node.selection_request:
            selection_request = self._node.selection_request
            self._node.selection_request = None  # 플래그 리셋

            self._node.get_logger().info(f'상품 선택 수신: product_id={selection_request.product_id}, bbox_number={selection_request.bbox_number}')

            # 선택된 상품 정보 저장
            self._node.selected_product_id = selection_request.product_id
            self._node.selected_bbox_number = selection_request.bbox_number

            # Vision 인식 결과에서 선택된 bbox_number에 해당하는 bbox_coords를 찾아 저장
            target_bbox_coords = None
            if hasattr(self._node, 'detection_result') and self._node.detection_result and self._node.detection_result.success:
                detected_products = self._node.detection_result.products
                self._node.get_logger().info(f'좌표를 찾기 전, detection_result에 들어있는 상품 개수: {len(detected_products)}')
                
                # 인식된 상품이 1개일 경우, bbox_number와 상관없이 해당 상품을 선택한 것으로 간주 (테스트 편의성)
                if len(detected_products) == 1:
                    target_bbox_coords = detected_products[0].bbox_coords
                    self._node.get_logger().info(f'인식된 상품이 1개이므로 BBox Number {detected_products[0].bbox_number}를 선택한 것으로 간주합니다.')
                # 인식된 상품이 여러 개일 경우, 요청된 bbox_number와 일치하는 것을 찾음
                else:
                    for product in detected_products:
                        if product.bbox_number == selection_request.bbox_number:
                            target_bbox_coords = product.bbox_coords
                            self._node.get_logger().info(f'BBox Number {selection_request.bbox_number}에 해당하는 좌표를 찾았습니다.')
                            break
            
            if target_bbox_coords is None:
                self._node.get_logger().error(f'BBox Number {selection_request.bbox_number}에 해당하는 좌표를 찾지 못했습니다.')

            self._node.selected_target_position = target_bbox_coords
            
            # PICKING_PRODUCT 상태로 전환
            from .picking_product import PickingProductState
            new_state = PickingProductState(self._node)
            self._node.state_machine.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('WAITING_SELECTION 상태 탈출')
