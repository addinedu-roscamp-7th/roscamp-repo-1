import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import (
    PickeeVisionDetection,
    PickeeVisionObstacles,
    PickeeVisionStaffLocation,
    PickeeVisionStaffRegister
)
from shopee_interfaces.srv import (
    PickeeVisionDetectProducts,
    PickeeVisionSetMode,
    PickeeVisionTrackStaff
)

class MockVisionNode(Node):
    def __init__(self):
        super().__init__('mock_vision_node')
        
        # Service Servers 생성
        self.detect_service = self.create_service(
            PickeeVisionDetectProducts,
            '/pickee/vision/detect_products',
            self.detect_products_callback
        )
        
        self.set_mode_service = self.create_service(
            PickeeVisionSetMode,
            '/pickee/vision/set_mode',
            self.set_mode_callback
        )
        
        self.track_staff_service = self.create_service(
            PickeeVisionTrackStaff,
            '/pickee/vision/track_staff',
            self.track_staff_callback
        )
        
        # Publishers 생성
        self.detection_pub = self.create_publisher(
            PickeeVisionDetection,
            '/pickee/vision/product_detection',
            10
        )
        
        self.obstacles_pub = self.create_publisher(
            PickeeVisionObstacles,
            '/pickee/vision/obstacles',
            10
        )
        
        self.staff_location_pub = self.create_publisher(
            PickeeVisionStaffLocation,
            '/pickee/vision/staff_location',
            10
        )
        
        self.staff_register_pub = self.create_publisher(
            PickeeVisionStaffRegister,
            '/pickee/vision/staff_register',
            10
        )
        
        # 주기적으로 장애물 정보 발행
        self.obstacles_timer = self.create_timer(2.0, self.publish_obstacles)
        
        self.get_logger().info('Mock Vision Node started successfully')
    
    def detect_products_callback(self, request, response):
        self.get_logger().info(f'Received detect products request: location_id={request.location_id}')
        
        # 1초 후에 제품 감지 메시지 발행
        self.create_timer(1.0, lambda: self.publish_detection(request.location_id))
        
        response.success = True
        response.message = 'Detection started'
        return response
    
    def set_mode_callback(self, request, response):
        self.get_logger().info(f'Received set mode request: mode={request.mode}')
        
        response.success = True
        response.message = f'Mode set to {request.mode}'
        return response
    
    def track_staff_callback(self, request, response):
        self.get_logger().info(f'Received track staff request: staff_id={request.staff_id}')
        
        # 2초 후에 직원 위치 메시지 발행
        self.create_timer(2.0, lambda: self.publish_staff_location(request.staff_id))
        
        response.success = True
        response.message = 'Staff tracking started'
        return response
    
    def publish_detection(self, location_id):
        msg = PickeeVisionDetection()
        msg.robot_id = 1
        msg.location_id = location_id
        msg.product_id = 'P001'
        msg.confidence = 0.95
        
        self.detection_pub.publish(msg)
        self.get_logger().info(f'Published detection: location_id={location_id}, product_id=P001')
    
    def publish_obstacles(self):
        msg = PickeeVisionObstacles()
        msg.robot_id = 1
        # 실제 메시지 구조에 맞게 필드를 설정하거나 기본값만 사용
        # obstacles_detected 필드를 사용하지 않고 기본 메시지만 발행
        
        self.obstacles_pub.publish(msg)
        # 로그도 간소화
        # self.get_logger().info('Published obstacles info')
    
    def publish_staff_location(self, staff_id):
        msg = PickeeVisionStaffLocation()
        msg.robot_id = 1
        msg.staff_id = staff_id
        # 실제 메시지 구조에 따라 location 필드 설정
        # msg.location.x = 2.0
        # msg.location.y = 3.0
        msg.confidence = 0.9
        
        self.staff_location_pub.publish(msg)
        self.get_logger().info(f'Published staff location: staff_id={staff_id}')

def main():
    rclpy.init()
    node = MockVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()