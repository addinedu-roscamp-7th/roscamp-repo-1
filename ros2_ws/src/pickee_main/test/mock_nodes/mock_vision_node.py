import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import (
    PickeeVisionDetection,
    PickeeVisionObstacles,
    PickeeVisionStaffLocation,
    PickeeVisionStaffRegister,
    PickeeVisionCartCheck
)
from shopee_interfaces.srv import (
    PickeeVisionDetectProducts,
    PickeeVisionSetMode,
    PickeeVisionTrackStaff,
    PickeeVisionCheckProductInCart,
    PickeeVisionCheckCartPresence,
    PickeeVisionVideoStreamStart,
    PickeeVisionVideoStreamStop,
    PickeeVisionRegisterStaff,
    PickeeTtsRequest
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
        
        self.check_product_in_cart_service = self.create_service(
            PickeeVisionCheckProductInCart,
            '/pickee/vision/check_product_in_cart',
            self.check_product_in_cart_callback
        )
        
        self.check_cart_presence_service = self.create_service(
            PickeeVisionCheckCartPresence,
            '/pickee/vision/check_cart_presence',
            self.check_cart_presence_callback
        )
        
        self.video_stream_start_service = self.create_service(
            PickeeVisionVideoStreamStart,
            '/pickee/vision/video_stream_start',
            self.video_stream_start_callback
        )
        
        self.video_stream_stop_service = self.create_service(
            PickeeVisionVideoStreamStop,
            '/pickee/vision/video_stream_stop',
            self.video_stream_stop_callback
        )
        
        self.register_staff_service = self.create_service(
            PickeeVisionRegisterStaff,
            '/pickee/vision/register_staff',
            self.register_staff_callback
        )
        
        # Publishers 생성 (명세서에 맞는 토픽 이름 사용)
        self.detection_pub = self.create_publisher(
            PickeeVisionDetection,
            '/pickee/vision/detection_result',
            10
        )
        
        self.obstacles_pub = self.create_publisher(
            PickeeVisionObstacles,
            '/pickee/vision/obstacle_detected',
            10
        )
        
        self.staff_location_pub = self.create_publisher(
            PickeeVisionStaffLocation,
            '/pickee/vision/staff_location',
            10
        )
        
        self.staff_register_pub = self.create_publisher(
            PickeeVisionStaffRegister,
            '/pickee/vision/register_staff_result',
            10
        )
        
        self.cart_check_pub = self.create_publisher(
            PickeeVisionCartCheck,
            '/pickee/vision/cart_check_result',
            10
        )
        
        # TTS 요청을 위한 Service Client (Vision에서 Main으로 TTS 요청)
        self.tts_request_client = self.create_client(
            PickeeTtsRequest,
            '/pickee/tts_request'
        )
        
        # 주기적으로 장애물 정보 발행
        self.obstacles_timer = self.create_timer(2.0, self.publish_obstacles)
        
        # 5초마다 TTS 요청 테스트 (선택적)
        # self.tts_timer = self.create_timer(5.0, lambda: self.test_tts_request())
        
        self.get_logger().info('Mock Vision Node started successfully')
    
    def test_tts_request(self):
        # TTS 요청 테스트 함수
        import asyncio
        asyncio.create_task(self.call_tts_request("테스트 음성 메시지입니다."))
    
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
    
    def check_product_in_cart_callback(self, request, response):
        self.get_logger().info(f'Received check product in cart request: product_id={request.product_id}')
        
        # 1초 후에 장바구니 확인 결과 발행
        self.create_timer(1.0, lambda: self.publish_cart_check_result(request.product_id))
        
        response.success = True
        response.message = 'Cart product check started'
        return response
    
    def check_cart_presence_callback(self, request, response):
        self.get_logger().info(f'Received check cart presence request')
        
        # 모의 장바구니 존재 확인 결과
        response.success = True
        response.cart_present = True
        response.message = 'Cart detected'
        return response
    
    def video_stream_start_callback(self, request, response):
        self.get_logger().info(f'Received video stream start request: user_type={request.user_type}, user_id={request.user_id}')
        
        response.success = True
        response.message = 'Video streaming started'
        return response
    
    def video_stream_stop_callback(self, request, response):
        self.get_logger().info(f'Received video stream stop request: user_type={request.user_type}, user_id={request.user_id}')
        
        response.success = True
        response.message = 'Video streaming stopped'
        return response
    
    def register_staff_callback(self, request, response):
        self.get_logger().info(f'Received register staff request')
        
        # 2초 후에 직원 등록 결과 발행
        self.create_timer(2.0, self.publish_staff_register_result)
        
        response.accepted = True
        response.message = 'Staff registration process accepted'
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
    
    def publish_cart_check_result(self, product_id):
        msg = PickeeVisionCartCheck()
        msg.robot_id = 1
        msg.order_id = 1
        msg.success = True
        msg.product_id = product_id
        msg.found = True  # 모의 테스트에서는 항상 찾았다고 가정
        msg.quantity = 2
        msg.message = "Product found in cart"
        
        self.cart_check_pub.publish(msg)
        self.get_logger().info(f'Published cart check result: product_id={product_id}, found=True')
    
    def publish_staff_register_result(self):
        msg = PickeeVisionStaffRegister()
        msg.robot_id = 1
        msg.success = True
        msg.message = "Staff registration successful"
        
        self.staff_register_pub.publish(msg)
        self.get_logger().info('Published staff register result: success=True')
    
    async def call_tts_request(self, text_to_speak):
        # Main Controller에 TTS 요청 전송 (테스트용)
        request = PickeeTtsRequest.Request()
        request.text_to_speak = text_to_speak
        
        if not self.tts_request_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('TTS request service not available')
            return False
        
        try:
            future = self.tts_request_client.call_async(request)
            response = await future
            self.get_logger().info(f'TTS request completed: success={response.success}')
            return response.success
        except Exception as e:
            self.get_logger().error(f'TTS request failed: {str(e)}')
            return False

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