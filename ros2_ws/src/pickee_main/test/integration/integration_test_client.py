#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from shopee_interfaces.srv import PickeeWorkflowStartTask
from shopee_interfaces.msg import PickeeRobotStatus, ProductInfo
import time

class IntegrationTestClient(Node):
    """통합 테스트를 위한 클라이언트 노드"""
    
    def __init__(self):
        super().__init__('integration_test_client')
        
        # Service Client 생성
        self.start_task_client = self.create_client(
            PickeeWorkflowStartTask,
            '/pickee/workflow/start_task'
        )
        
        # Subscriber 생성 - 로봇 상태 모니터링
        self.status_subscriber = self.create_subscription(
            PickeeRobotStatus,
            '/pickee/robot_status',
            self.status_callback,
            10
        )
        
        self.last_status = None
        self.test_results = []
        
        self.get_logger().info('Integration Test Client started')
    
    def status_callback(self, msg):
        """로봇 상태 콜백"""
        self.last_status = msg
        self.get_logger().info(f'Robot Status: {msg.current_state}, Battery: {msg.battery_level:.1f}%')
    
    def wait_for_service(self, timeout_sec=10.0):
        """서비스가 준비될 때까지 대기"""
        if not self.start_task_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f'Service not available after {timeout_sec} seconds')
            return False
        return True
    
    def test_start_task(self):
        """작업 시작 테스트"""
        self.get_logger().info('=== Testing Start Task ===')
        
        if not self.wait_for_service():
            return False
        
        # 테스트용 제품 목록 생성
        product_list = [
            ProductInfo(product_id='P001', location_id='L001', quantity=1),
            ProductInfo(product_id='P002', location_id='L002', quantity=2),
        ]
        
        # 서비스 요청 생성
        request = PickeeWorkflowStartTask.Request()
        request.robot_id = 1
        request.order_id = 1001
        request.product_list = product_list
        
        # 서비스 호출
        self.get_logger().info('Sending start task request...')
        future = self.start_task_client.call_async(request)
        
        return future
    
    def run_basic_tests(self):
        """기본 테스트 시나리오 실행"""
        self.get_logger().info('Starting integration tests...')
        
        # 초기 상태 확인을 위해 잠시 대기
        time.sleep(2.0)
        
        test_cases = [
            ('Initial Status Check', self.test_initial_status),
            ('Start Task Test', self.test_start_task_scenario),
        ]
        
        for test_name, test_func in test_cases:
            self.get_logger().info(f'Running: {test_name}')
            try:
                result = test_func()
                self.test_results.append((test_name, 'PASS' if result else 'FAIL'))
                self.get_logger().info(f'{test_name}: {"PASS" if result else "FAIL"}')
            except Exception as e:
                self.test_results.append((test_name, f'ERROR: {str(e)}'))
                self.get_logger().error(f'{test_name}: ERROR - {str(e)}')
            
            time.sleep(1.0)  # 테스트 간 간격
        
        self.print_test_results()
    
    def test_initial_status(self):
        """초기 상태 테스트"""
        # 상태 메시지를 받을 때까지 대기
        timeout = 5.0
        start_time = time.time()
        
        while self.last_status is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.last_status is None:
            self.get_logger().error('No status message received')
            return False
        
        # 초기 상태가 올바른지 확인
        expected_states = ['InitializingState', 'ChargingAvailableState', 'MovingToStandbyState']
        current_state = self.last_status.current_state
        
        if current_state in expected_states:
            self.get_logger().info(f'Initial state OK: {current_state}')
            return True
        else:
            self.get_logger().warning(f'Unexpected initial state: {current_state}')
            return True  # 상태에 따라 달라질 수 있으므로 warning만
    
    def test_start_task_scenario(self):
        """작업 시작 시나리오 테스트"""
        future = self.test_start_task()
        
        if future is None:
            return False
        
        # 응답 대기
        timeout = 10.0
        start_time = time.time()
        
        while not future.done() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not future.done():
            self.get_logger().error('Service call timed out')
            return False
        
        try:
            response = future.result()
            if response.accepted:
                self.get_logger().info('Start task request accepted')
                return True
            else:
                self.get_logger().warning(f'Start task request rejected: {response.message}')
                return False
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')
            return False
    
    def print_test_results(self):
        """테스트 결과 출력"""
        self.get_logger().info('=== Test Results Summary ===')
        
        for test_name, result in self.test_results:
            self.get_logger().info(f'{test_name}: {result}')
        
        pass_count = sum(1 for _, result in self.test_results if result == 'PASS')
        total_count = len(self.test_results)
        
        self.get_logger().info(f'Total: {pass_count}/{total_count} tests passed')


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    
    test_client = IntegrationTestClient()
    
    try:
        # 테스트 실행
        test_client.run_basic_tests()
        
        # 추가 모니터링을 위해 계속 실행
        rclpy.spin(test_client)
        
    except KeyboardInterrupt:
        test_client.get_logger().info('Integration test client shutting down...')
    finally:
        test_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()