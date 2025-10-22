
import asyncio
import json
import unittest
import threading
from typing import Any, Dict, List

import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import launch
import launch_testing
from launch.actions import ExecuteProcess

from main_service.client_utils import MainServiceClient
from main_service.database_models import Customer, Order, OrderItem, Product, Section, Shelf, Location
from main_service.config import settings


# E2E 테스트에 필요한 노드들을 정의하고 실행하는 부분
@pytest.mark.launch_test
def generate_test_description():
    # 테스트 전에 DB가 초기화되도록 간단한 스크립트 실행도 가능
    # 여기서는 설명을 위해 생략

    # 1. main_service 노드 실행
    main_service_node = ExecuteProcess(
        cmd=['ros2', 'run', 'main_service', 'main_service_node'],
        output='screen',
        name='main_service'
    )
    
    # 2. 가짜 Pickee 로봇 노드 실행
    # E2E 테스트에서는 pickee만 시뮬레이션
    mock_pickee_node = ExecuteProcess(
        cmd=['ros2', 'run', 'main_service', 'mock_pickee_node'],
        output='screen',
        name='mock_pickee'
    )

    return launch.LaunchDescription([
        main_service_node,
        mock_pickee_node,
        # 모든 노드가 준비될 때까지 기다렸다가 테스트를 시작하라는 신호
        launch_testing.actions.ReadyToTest()
    ])


# 실제 테스트 로직을 담고 있는 클래스
# unittest.TestCase를 상속받아 풍부한 assert 메서드 활용
class TestFullWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 테스트 전체 시작 시 한 번만 ROS 초기화
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        # 테스트 전체 종료 시 한 번만 ROS 종료
        rclpy.shutdown()

    def setUp(self):
        # 각 테스트 케이스 시작 전, 테스트용 노드 및 클라이언트 생성
        self.test_node = Node('e2e_test_client')
        self.feedback_messages: List[str] = []
        
        # mock 노드가 발행하는 피드백을 수신할 구독자
        self.feedback_sub = self.test_node.create_subscription(
            String,
            '/test/feedback',
            self.feedback_callback,
            10
        )
        
        # ROS 노드를 별도 스레드에서 실행
        self.ros_spin_thread = threading.Thread(target=rclpy.spin, args=(self.test_node,))
        self.ros_spin_thread.daemon = True
        self.ros_spin_thread.start()

    def tearDown(self):
        # 각 테스트 케이스 종료 후, 노드 소멸
        self.test_node.destroy_node()

    def feedback_callback(self, msg: String):
        """/test/feedback 토픽 메시지를 수신하면 리스트에 저장"""
        self.test_node.get_logger().info(f"Received feedback: {msg.data}")
        self.feedback_messages.append(msg.data)

    async def wait_for_feedback(self, expected_prefix: str, timeout: float = 5.0) -> str:
        """특정 피드백 메시지가 도착할 때까지 대기"""
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            for msg in self.feedback_messages:
                if msg.startswith(expected_prefix):
                    return msg
            await asyncio.sleep(0.1)
        raise TimeoutError(f"'{expected_prefix}' 피드백을 시간 내에 받지 못했습니다.")

    async def run_test_scenario(self):
        """실제 테스트 시나리오를 실행하는 비동기 함수"""
        # 노드가 완전히 시작될 때까지 잠시 대기
        await asyncio.sleep(2)

        # main_service와 통신할 TCP 클라이언트 생성
        tcp_client = MainServiceClient(host=settings.API_HOST, port=settings.API_PORT)
        await tcp_client.connect()

        try:
            # Wait until the mock robot reports as IDLE
            self.test_node.get_logger().info("E2E TEST: Waiting for mock robot to become available...")
            is_robot_available = False
            wait_timeout = 10.0 # 10초 대기
            start_wait_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_wait_time < wait_timeout:
                status_res = await tcp_client.send_request('robot_status_request', {})
                if status_res.get('result'):
                    robots = status_res.get('data', {}).get('robots', [])
                    for robot in robots:
                        # mock_pickee_node의 기본 ID는 1
                        if robot.get('robot_id') == 1 and robot.get('status') == 'IDLE':
                            is_robot_available = True
                            break
                if is_robot_available:
                    self.test_node.get_logger().info("E2E TEST: Mock robot is available!")
                    break
                await asyncio.sleep(0.5)

            self.assertTrue(is_robot_available, "Mock robot did not become available in time.")

            # 1. (Given) 테스트용 주문 데이터 준비
            # 실제 DB에 테스트용 사용자와 상품이 있다는 가정 하에 진행
            # 혹은 테스트 시작 전 DB를 초기화하는 스크립트 필요
            user_id = 'admin'
            cart_items = [{'product_id': 1, 'quantity': 1}]
            order_payload = {
                'user_id': user_id,
                'cart_items': cart_items,
            }

            # 2. (When) 가짜 앱이 main_service에 TCP로 주문 생성 요청
            self.test_node.get_logger().info("E2E TEST: Sending order_create request...")
            response = await tcp_client.send_request('order_create', order_payload)
            self.test_node.get_logger().info(f"E2E TEST: Received response: {response}")

            # 3. (Then) main_service가 정상 응답을 주었는지 확인
            self.assertTrue(response.get('result'), "Order creation failed in response")
            order_id = response.get('data', {}).get('order_id')
            self.assertIsNotNone(order_id, "Order ID not in response")

            # 4. (Then) mock_pickee_node가 start_task 서비스 호출을 받았는지 피드백 토픽으로 확인
            self.test_node.get_logger().info("E2E TEST: Waiting for mock node feedback...")
            feedback = await self.wait_for_feedback(f'start_task_called:{order_id}')
            
            # 5. (Then) 피드백 메시지가 정확한지 최종 검증
            self.assertEqual(feedback, f'start_task_called:{order_id}')
            self.test_node.get_logger().info("E2E TEST: Success! Mock node received the task.")

        finally:
            await tcp_client.close()

    def test_order_creation_happy_path(self):
        """
        [E2E-TC-01] 주문 생성 Happy Path 시나리오
        사용자가 주문을 생성하면, main_service가 이를 처리하여
        pickee 로봇에게 start_task 명령을 내리는지 검증한다.
        """
        # 비동기 테스트 시나리오를 이벤트 루프에서 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_test_scenario())
        finally:
            loop.close()

