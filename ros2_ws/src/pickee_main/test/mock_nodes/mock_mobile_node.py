import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobileArrival, PickeeMobilePose, PickeeMobileSpeedControl
from shopee_interfaces.srv import PickeeMobileMoveToLocation, PickeeMobileUpdateGlobalPath
import threading
import time

class MockMobileNode(Node):
    def __init__(self):
        super().__init__('mock_mobile_node')
        
        # Service Server 생성
        self.move_service = self.create_service(
            PickeeMobileMoveToLocation,
            '/pickee/mobile/move_to_location',
            self.move_to_location_callback
        )
        
        self.update_global_path_service = self.create_service(
            PickeeMobileUpdateGlobalPath,
            '/pickee/mobile/update_global_path',
            self.update_global_path_callback
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
        
        # Subscriber 생성 (속도 제어 명령 수신)
        self.speed_control_sub = self.create_subscription(
            PickeeMobileSpeedControl,
            '/pickee/mobile/speed_control',
            self.speed_control_callback,
            10
        )
        
        # 상태 변수
        self.is_moving = False
        self.current_speed_mode = "normal"
        self.target_speed = 1.0
        self.global_path = []
        
        # 주기적으로 위치 정보 발행
        self.pose_timer = self.create_timer(1.0, self.publish_pose)
        
        self.get_logger().info('Mock Mobile Node started successfully')
    
    def move_to_location_callback(self, request, response):
        self.get_logger().info(f'Received move request: location_id={request.location_id}')
        
        self.is_moving = True
        
        # 2초 후에 도착 메시지를 한 번만 발행하도록 스레드 사용
        threading.Thread(
            target=self.delayed_arrival_publish,
            args=(request.location_id,),
            daemon=True
        ).start()
        
        response.success = True
        response.message = 'Move command accepted'
        return response
    
    def delayed_arrival_publish(self, location_id):
        # 2초 대기 후 한 번만 도착 메시지 발행
        time.sleep(2.0)
        self.publish_arrival(location_id)
    
    def publish_arrival(self, location_id):
        msg = PickeeMobileArrival()
        msg.robot_id = 1
        msg.order_id = 123  # 테스트용 order_id
        msg.location_id = location_id
        msg.final_pose.x = 10.5  # 테스트용 최종 위치
        msg.final_pose.y = 5.2
        msg.final_pose.theta = 1.57
        msg.position_error = 0.05  # 5cm 오차
        msg.travel_time = 15.5  # 15.5초 이동 시간
        msg.message = f"Successfully arrived at location {location_id}"
        
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
    
    def update_global_path_callback(self, request, response):
        # 전역 경로 업데이트 서비스 콜백
        self.get_logger().info(f'Received update global path request: location_id={request.location_id}, path_length={len(request.global_path)}')
        
        # 전역 경로 저장
        self.global_path = request.global_path
        
        # 모의 응답
        response.success = True
        response.message = f'Global path updated with {len(request.global_path)} waypoints'
        return response
    
    def speed_control_callback(self, msg):
        # 속도 제어 명령 수신 콜백
        self.get_logger().info(f'Received speed control: mode={msg.speed_mode}, target_speed={msg.target_speed}, reason={msg.reason}')
        
        # 속도 제어 상태 업데이트
        self.current_speed_mode = msg.speed_mode
        self.target_speed = msg.target_speed
        
        # 장애물 정보가 있으면 로그 출력
        if msg.obstacles:
            self.get_logger().info(f'Obstacles detected: {len(msg.obstacles)} obstacles')
            for i, obstacle in enumerate(msg.obstacles):
                self.get_logger().info(f'  Obstacle {i+1}: type={obstacle.obstacle_type}, distance={obstacle.distance}m')

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