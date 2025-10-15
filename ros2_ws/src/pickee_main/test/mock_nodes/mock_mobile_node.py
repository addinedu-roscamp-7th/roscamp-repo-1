import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobileArrival, PickeeMobilePose
from shopee_interfaces.srv import PickeeMobileMoveToLocation

class MockMobileNode(Node):
    def __init__(self):
        super().__init__('mock_mobile_node')
        
        # Service Server 생성
        self.move_service = self.create_service(
            PickeeMobileMoveToLocation,
            '/pickee/mobile/move_to_location',
            self.move_to_location_callback
        )
        
        # Publisher 생성
        self.arrival_pub = self.create_publisher(
            PickeeMobileArrival,
            '/pickee/mobile/arrival',
            10
        )
        
        self.pose_pub = self.create_publisher(
            PickeeMobilePose,
            '/pickee/mobile/pose',
            10
        )
        
        # 상태 변수
        self.is_moving = False
        
        # 주기적으로 위치 정보 발행
        self.pose_timer = self.create_timer(1.0, self.publish_pose)
        
        self.get_logger().info('Mock Mobile Node started successfully')
    
    def move_to_location_callback(self, request, response):
        self.get_logger().info(f'Received move request: location_id={request.location_id}')
        
        self.is_moving = True
        
        # 2초 후에 도착 메시지 발행하도록 타이머 설정
        self.create_timer(2.0, lambda: self.publish_arrival(request.location_id))
        
        response.success = True
        response.message = 'Move command accepted'
        return response
    
    def publish_arrival(self, location_id):
        msg = PickeeMobileArrival()
        msg.robot_id = 1
        msg.location_id = location_id
        msg.success = True
        
        self.arrival_pub.publish(msg)
        self.is_moving = False
        self.get_logger().info(f'Published arrival: location_id={location_id}')
    
    def publish_pose(self):
        msg = PickeeMobilePose()
        msg.robot_id = 1
        msg.current_pose.x = 1.0
        msg.current_pose.y = 2.0
        msg.current_pose.theta = 0.5
        msg.battery_level = 80.0
        # is_moving 필드 제거 (메시지에 없는 필드)
        
        self.pose_pub.publish(msg)

def main():
    rclpy.init()
    node = MockMobileNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()