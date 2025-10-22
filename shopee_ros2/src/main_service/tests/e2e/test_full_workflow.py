
import asyncio
import json
import os
import threading
import unittest
from typing import Any, Dict, List

import launch
import launch_testing
import pytest
import rclpy
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from rclpy.node import Node
from std_msgs.msg import String

from main_service.client_utils import MainServiceClient
from main_service.database_models import Customer, Order, OrderItem, Product
from main_service.database_manager import DatabaseManager
from main_service.config import settings


# E2E 테스트에 필요한 노드들을 정의하고 실행하는 부분
@pytest.mark.launch_test
def generate_test_description():
    # 테스트 전에 DB가 초기화되도록 간단한 스크립트 실행도 가능
    # 여기서는 설명을 위해 생략

    # 1. main_service 노드 실행
    api_port = LaunchConfiguration('api_port')
    app_host = LaunchConfiguration('app_host')

    main_service_node = ExecuteProcess(
        cmd=[
            'ros2',
            'run',
            'main_service',
            'main_service_node',
        ],
        additional_env={
            'SHOPEE_API_PORT': api_port,
            'SHOPEE_API_HOST': app_host,
        },
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
        launch.actions.DeclareLaunchArgument('api_port', default_value='5100'),
        launch.actions.DeclareLaunchArgument('app_host', default_value='127.0.0.1'),
        SetEnvironmentVariable('SHOPEE_API_PORT', api_port),
        SetEnvironmentVariable('SHOPEE_API_HOST', app_host),
        main_service_node,
        mock_pickee_node,
        launch_testing.actions.ReadyToTest(),
    ])


# 실제 테스트 로직을 담고 있는 클래스
# unittest.TestCase를 상속받아 풍부한 assert 메서드 활용
class TestFullWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 테스트 전체 시작 시 한 번만 ROS 초기화
        rclpy.init()
        cls._db_manager = DatabaseManager()
        cls._seed_test_data()

    @classmethod
    def tearDownClass(cls):
        # 테스트 전체 종료 시 한 번만 ROS 종료 및 테스트 데이터 정리
        cls._clear_test_data()
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
        self._ros_spin_stop = threading.Event()

        def _spin():
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(self.test_node)
            while rclpy.ok() and not self._ros_spin_stop.is_set():
                executor.spin_once(timeout_sec=0.1)
            executor.remove_node(self.test_node)

        self.ros_spin_thread = threading.Thread(target=_spin)
        self.ros_spin_thread.daemon = True
        self.ros_spin_thread.start()

    def tearDown(self):
        # 각 테스트 케이스 종료 후, 노드 및 스레드 종료
        self._ros_spin_stop.set()
        if self.ros_spin_thread.is_alive():
            self.ros_spin_thread.join(timeout=5.0)
        self.test_node.destroy_node()

    @classmethod
    def _seed_test_data(cls):
        with cls._db_manager.session_scope() as session:
            # 사용자 존재 여부 확인 후 생성
            if not session.query(Customer).filter_by(customer_id='admin').first():
                session.add(Customer(
                    customer_id='admin',
                    password_hash='dummy',
                    name='Admin',
                    address='Test Address',
                ))

            if not session.query(Product).filter_by(product_id=99901).first():
                session.add(Product(
                    product_id=99901,
                    barcode='TEST-99901',
                    name='E2E Test Product',
                    quantity=10,
                    price=1000,
                    section_id=1,
                    category='test',
                    allergy_info_id=1,
                    is_vegan_friendly=True,
                ))

    @classmethod
    def _clear_test_data(cls):
        with cls._db_manager.session_scope() as session:
            session.query(OrderItem).filter(OrderItem.product_id == 99901).delete()
            session.query(Order).filter(Order.customer_id == 'admin').delete()
            session.query(Product).filter(Product.product_id == 99901).delete()

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
        app_host = os.environ.get('SHOPEE_API_HOST', settings.API_HOST)
        app_port = int(os.environ.get('SHOPEE_API_PORT', settings.API_PORT))
        tcp_client = MainServiceClient(host=app_host, port=app_port)
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
            user_id = 'admin'
            cart_items = [{'product_id': 99901, 'quantity': 1}]
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
