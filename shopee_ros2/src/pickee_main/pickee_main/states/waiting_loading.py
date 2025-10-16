from .state import State


class WaitingLoadingState(State):
    # 적재대기중 상태
    
    def on_enter(self):
        self._node.get_logger().info('WAITING_LOADING 상태 진입')
        self.loading_completed = False
        # 직원이 상품을 장바구니에 싣는 것을 대기
        self._node.get_logger().info('직원이 상품을 장바구니에 적재하는 것을 기다리고 있습니다.')
        
    def execute(self):
        # 적재 완료를 위한 대기 로직
        # Vision을 통해 장바구니 내 상품 확인이나 직원의 신호를 대기
        
        # 임시로 시간 기반 완료 처리 (실제로는 Vision이나 사용자 입력 필요)
        if not hasattr(self, 'loading_start_time'):
            import time
            self.loading_start_time = time.time()
        
        # 10초 후 자동으로 적재 완료로 간주 (테스트용)
        import time
        if time.time() - self.loading_start_time > 10.0:
            self._node.get_logger().info('상품 적재가 완료되었습니다.')
            
            # 적재 완료 보고 발행
            # 현재 처리 중인 상품 ID 사용
            current_product_id = 1  # 기본값
            if hasattr(self._node, 'target_product_ids') and self._node.target_product_ids:
                current_product_id = self._node.target_product_ids[0]
            
            self._node.publish_product_loaded(
                product_id=current_product_id,
                quantity=1,
                success=True,
                message='Product loading completed'
            )
            
            # 다음 매대로 이동하거나 직원을 추종
            from .following_staff import FollowingStaffState
            new_state = FollowingStaffState(self._node)
            self.transition_to(new_state)
    
    def on_exit(self):
        self._node.get_logger().info('WAITING_LOADING 상태 탈출')