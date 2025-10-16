from .state import State
from shopee_interfaces.srv import (
    PickeeArmMoveToPose,
    PickeeVisionSetMode,
    PickeeVisionDetectProducts
)

class DetectingProductState(State):
    # 상품인식중 상태
    
    def on_enter(self):
        self._node.get_logger().info('DETECTING_PRODUCT 상태 진입')
        self.phase = 0
        self.future = None

        # 0단계: Arm을 shelf_view 자세로 변경 시작
        self._node.get_logger().info('0단계: Arm을 shelf_view 자세로 변경 시작')
        request = PickeeArmMoveToPose.Request()
        request.robot_id = self._node.robot_id
        request.order_id = self._node.current_order_id
        request.pose_type = 'shelf_view'
        self.future = self._node.arm_move_to_pose_client.call_async(request)

    def execute(self):
        if self.future and self.future.done():
            result = self.future.result()
            if not result or not result.success:
                self._node.get_logger().error(f'{self.phase}단계 실패. 더 이상 진행하지 않음.')
                # 실패 시 CHARGING_AVAILABLE 상태로 복귀
                from .charging_available import ChargingAvailableState
                new_state = ChargingAvailableState(self._node)
                self.transition_to(new_state)
                self.future = None
                return

            # 이전 단계 성공, 다음 단계로 진행
            if self.phase == 0: # Arm 이동 완료
                self.phase = 1
                self._node.get_logger().info('1단계: Vision 모드를 detect_products로 설정')
                request = PickeeVisionSetMode.Request()
                request.robot_id = self._node.robot_id
                request.mode = 'detect_products'
                self.future = self._node.vision_set_mode_client.call_async(request)
            
            elif self.phase == 1: # Vision 모드 설정 완료
                self.phase = 2
                target_product_ids = getattr(self._node, 'target_product_ids', [1, 2, 3])
                self._node.get_logger().info(f'2단계: 상품 인식 시작: {target_product_ids}')
                request = PickeeVisionDetectProducts.Request()
                request.robot_id = self._node.robot_id
                request.order_id = self._node.current_order_id
                request.product_ids = target_product_ids
                self.future = self._node.vision_detect_products_client.call_async(request)

            elif self.phase == 2: # 상품 인식 명령 전송 완료
                self.phase = 3
                self.future = None # 모든 서비스 호출 완료
                self._node.get_logger().info('3단계: 모든 사전 서비스 호출 완료. 인식 토픽 대기중.')

        # 3단계: /pickee/vision/detection_result 토픽 수신 확인
        if self.phase == 3:
            if hasattr(self._node, 'detection_result') and self._node.detection_result:
                detection_result = self._node.detection_result
                self._node.detection_result = None  # 플래그 리셋
                
                # 인식 결과를 Main Service에 보고
                self._node.publish_product_detected(detection_result.products)
                
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
        self._node.get_logger().info('DETECTING_PRODUCT 상태 탈출')