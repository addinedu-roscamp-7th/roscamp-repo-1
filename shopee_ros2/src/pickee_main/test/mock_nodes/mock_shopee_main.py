
import rclpy
from rclpy.node import Node
import threading
import time

# Import all the necessary message and service types
from shopee_interfaces.msg import (
    PickeeRobotStatus,
    PickeeArrival,
    PickeeProductDetection,
    PickeeCartHandover,
    PickeeProductSelection,
    PickeeMoveStatus,
    PickeeProductLoaded,
    Pose2D
)
from shopee_interfaces.srv import (
    PickeeWorkflowStartTask,
    PickeeWorkflowMoveToSection,
    PickeeProductDetect,
    PickeeProductProcessSelection,
    PickeeWorkflowEndShopping,
    PickeeWorkflowMoveToPackaging,
    PickeeWorkflowReturnToBase,
    PickeeWorkflowReturnToStaff,
    PickeeMainVideoStreamStart,
    PickeeMainVideoStreamStop,
    MainGetProductLocation,
    MainGetLocationPose,
    MainGetWarehousePose,
    MainGetSectionPose
)



class MockShopeeMain(Node):
    def __init__(self):
        super().__init__('mock_shopee_main')
        self.get_logger().info('Mock Shopee Main node started')

        # Subscribers to topics from pickee_main
        self.create_subscription(PickeeRobotStatus, '/pickee/robot_status', self.robot_status_callback, 10)
        self.create_subscription(PickeeArrival, '/pickee/arrival_notice', self.arrival_notice_callback, 10)
        self.create_subscription(PickeeProductDetection, '/pickee/product_detected', self.product_detected_callback, 10)
        self.create_subscription(PickeeCartHandover, '/pickee/cart_handover_complete', self.cart_handover_callback, 10)
        self.create_subscription(PickeeProductSelection, '/pickee/product/selection_result', self.product_selection_callback, 10)
        self.create_subscription(PickeeMoveStatus, '/pickee/moving_status', self.moving_status_callback, 10)
        self.create_subscription(PickeeProductLoaded, '/pickee/product/loaded', self.product_loaded_callback, 10)

        # Service servers for services called by pickee_main
        self.create_service(MainGetProductLocation, '/main/get_product_location', self.get_product_location_callback)
        self.create_service(MainGetLocationPose, '/main/get_location_pose', self.get_location_pose_callback)
        self.create_service(MainGetWarehousePose, '/main/get_warehouse_pose', self.get_warehouse_pose_callback)
        self.create_service(MainGetSectionPose, '/main/get_section_pose', self.get_section_pose_callback)

        # Service clients for services provided by pickee_main
        self.start_task_client = self.create_client(PickeeWorkflowStartTask, '/pickee/workflow/start_task')
        self.move_to_section_client = self.create_client(PickeeWorkflowMoveToSection, '/pickee/workflow/move_to_section')
        self.product_detect_client = self.create_client(PickeeProductDetect, '/pickee/product/detect')
        self.process_selection_client = self.create_client(PickeeProductProcessSelection, '/pickee/product/process_selection')
        self.end_shopping_client = self.create_client(PickeeWorkflowEndShopping, '/pickee/workflow/end_shopping')
        self.move_to_packaging_client = self.create_client(PickeeWorkflowMoveToPackaging, '/pickee/workflow/move_to_packaging')
        self.return_to_base_client = self.create_client(PickeeWorkflowReturnToBase, '/pickee/workflow/return_to_base')
        self.return_to_staff_client = self.create_client(PickeeWorkflowReturnToStaff, '/pickee/workflow/return_to_staff')
        self.video_start_client = self.create_client(PickeeMainVideoStreamStart, '/pickee/video_stream/start')
        self.video_stop_client = self.create_client(PickeeMainVideoStreamStop, '/pickee/video_stream/stop')

    # Subscriber callbacks
    def robot_status_callback(self, msg):
        self.get_logger().info(f'Received robot status: {msg}')

    def arrival_notice_callback(self, msg):
        self.get_logger().info(f'Received arrival notice: {msg}')

    def product_detected_callback(self, msg):
        self.get_logger().info(f'Received product detection: {msg}')

    def cart_handover_callback(self, msg):
        self.get_logger().info(f'Received cart handover: {msg}')

    def product_selection_callback(self, msg):
        self.get_logger().info(f'Received product selection: {msg}')

    def moving_status_callback(self, msg):
        self.get_logger().info(f'Received moving status: {msg}')

    def product_loaded_callback(self, msg):
        self.get_logger().info(f'Received product loaded: {msg}')

    # Service server callbacks
    def get_product_location_callback(self, request, response):
        self.get_logger().info(f'Received get product location request: {request}')
        response.success = True
        response.warehouse_id = 1
        response.section_id = 1
        return response

    def get_location_pose_callback(self, request, response):
        self.get_logger().info(f'Received get location pose request: {request}')
        response.success = True
        response.pose = Pose2D(x=1.0, y=1.0, theta=0.0)
        return response

    def get_warehouse_pose_callback(self, request, response):
        self.get_logger().info(f'Received get warehouse pose request: {request}')
        response.success = True
        response.pose = Pose2D(x=2.0, y=2.0, theta=0.0)
        return response

    def get_section_pose_callback(self, request, response):
        self.get_logger().info(f'Received get section pose request: {request}')
        response.success = True
        response.pose = Pose2D(x=3.0, y=3.0, theta=0.0)
        return response

    # Service client calls
    def call_start_task(self):
        request = PickeeWorkflowStartTask.Request()
        request.robot_id = 1
        request.order_id = 123
        self.start_task_client.call_async(request)
        self.get_logger().info('Called start task service')

def main(args=None):
    rclpy.init(args=args)
    mock_shopee_main = MockShopeeMain()
    rclpy.spin(mock_shopee_main)
    mock_shopee_main.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
