
import rclpy
from rclpy.node import Node
import time
import random

from shopee_interfaces.srv import PickeeVisionDetectProducts, PickeeVisionCheckProductInCart, PickeeVisionCheckCartPresence
from shopee_interfaces.msg import PickeeVisionDetection, PickeeDetectedProduct, BBox, PickeeVisionCartCheck

class ProductDetectorNode(Node):
    """
    상품 인식 및 장바구니 확인 요청을 처리하고, 그 결과를 토픽으로 발행하는 노드입니다.
    """
    def __init__(self):
        super().__init__('product_detector_node')

        # 서비스 서버 생성
        self.detect_products_srv = self.create_service(
            PickeeVisionDetectProducts, '/pickee/vision/detect_products', self.detect_products_callback)
        
        self.check_product_srv = self.create_service(
            PickeeVisionCheckProductInCart, '/pickee/vision/check_product_in_cart', self.check_product_in_cart_callback)

        self.check_cart_presence_srv = self.create_service(
            PickeeVisionCheckCartPresence, '/pickee/vision/check_cart_presence', self.check_cart_presence_callback)

        # 퍼블리셔 생성
        self.detection_result_pub = self.create_publisher(
            PickeeVisionDetection, '/pickee/vision/detection_result', 10)
        
        self.cart_check_result_pub = self.create_publisher(
            PickeeVisionCartCheck, '/pickee/vision/cart_check_result', 10)

        self.get_logger().info('Product Detector Node has been started.')

    def detect_products_callback(self, request, response):
        """/pickee/vision/detect_products 서비스 콜백"""
        self.get_logger().info(
            f'Detect products request received: order_id={request.order_id}, products={request.product_ids}')
        
        # 가짜 AI 처리 시간 (1.5초)
        time.sleep(1.5)

        # 가짜 탐지 결과 생성 및 발행
        msg = PickeeVisionDetection()
        msg.robot_id = request.robot_id
        msg.order_id = request.order_id
        msg.success = True
        
        for i, product_id in enumerate(request.product_ids):
            product = PickeeDetectedProduct()
            product.product_id = product_id
            product.bbox_number = i + 1
            product.bbox_coords = BBox(x1=100 + i*150, y1=150, x2=200 + i*150, y2=250)
            product.confidence = random.uniform(0.9, 0.99)
            msg.products.append(product)

        msg.message = f"{len(msg.products)} products detected."
        self.detection_result_pub.publish(msg)
        self.get_logger().info('Published detection result.')

        response.success = True
        response.message = "Detection started"
        return response

    def check_product_in_cart_callback(self, request, response):
        """/pickee/vision/check_product_in_cart 서비스 콜백"""
        self.get_logger().info(
            f'Check product in cart request: order_id={request.order_id}, product_id={request.product_id}')
        
        # 가짜 AI 처리 시간 (0.5초)
        time.sleep(0.5)

        # 가짜 확인 결과 생성 및 발행
        msg = PickeeVisionCartCheck()
        msg.robot_id = request.robot_id
        msg.order_id = request.order_id
        msg.success = True
        msg.product_id = request.product_id
        msg.found = True # 항상 찾았다고 가정
        msg.quantity = random.randint(1, 3)
        msg.message = "Product found in cart"
        self.cart_check_result_pub.publish(msg)
        self.get_logger().info('Published cart check result.')

        response.success = True
        response.message = "Cart product check started"
        return response

    def check_cart_presence_callback(self, request, response):
        """/pickee/vision/check_cart_presence 서비스 콜백"""
        self.get_logger().info(f'Check cart presence request received for order_id={request.order_id}')
        
        # 가짜 AI 처리 시간
        time.sleep(0.5)

        # 가짜 데이터 반환 (항상 장바구니가 있다고 가정)
        response.success = True
        response.cart_present = True
        response.message = "Cart detected"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ProductDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
