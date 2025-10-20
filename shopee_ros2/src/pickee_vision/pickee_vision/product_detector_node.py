import rclpy
from rclpy.node import Node
import cv2
import os

# YOLOv8 감지기 라이브러리
from .yolo_detector import YoloDetector

# 서비스 및 메시지 타입
from shopee_interfaces.srv import PickeeVisionDetectProducts, PickeeVisionCheckProductInCart, VisionCheckCartPresence
from shopee_interfaces.msg import (
    PickeeVisionDetection, 
    PickeeVisionCartCheck, 
    PickeeDetectedProduct, 
    BBox, 
    Point2D,
    DetectionInfo
)

# 임시 조치: shopee_interfaces 수정 전까지 테스트를 위한 가짜 클래스
class DetectionInfo:
    def __init__(self):
        self.polygon = []
        self.bbox_coords = None

# PickeeDetectedProduct가 필요한 필드를 갖도록 임시 수정
PickeeDetectedProduct.bbox_number = 0
PickeeDetectedProduct.detection_info = None

class ProductDetectorNode(Node):
    """
    YOLOv8 모델을 사용하여 상품을 인식하고, 그 결과를 ROS2 인터페이스에 맞게 발행합니다.
    """
    def __init__(self):
        super().__init__('product_detector_node')

        # YOLO 감지기 초기화
        model_path = os.path.join(os.path.dirname(__file__), 'yolov8s-seg.pt')
        try:
            self.detector = YoloDetector(model_path)
        except FileNotFoundError as e:
            self.get_logger().error(f"YOLO 모델 파일 로드 실패: {e}")
            raise e

        # 카메라 초기화 (Plan에 따라 카메라 인덱스 1 사용)
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.get_logger().error("카메라 인덱스 1을 열 수 없습니다.")
            raise IOError("Cannot open camera 1")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 서비스 서버 생성
        self.create_service(PickeeVisionDetectProducts, '/pickee/vision/detect_products', self.detect_products_callback)
        self.create_service(PickeeVisionCheckProductInCart, '/pickee/vision/check_product_in_cart', self.check_product_in_cart_callback)
        self.create_service(VisionCheckCartPresence, '/pickee/vision/check_cart_presence', self.check_cart_presence_callback)

        # 퍼블리셔 생성
        self.detection_result_pub = self.create_publisher(PickeeVisionDetection, '/pickee/vision/detection_result', 10)
        self.cart_check_result_pub = self.create_publisher(PickeeVisionCartCheck, '/pickee/vision/cart_check_result', 10)

        self.get_logger().info('Product Detector Node with YOLOv8 has been started.')

    def detect_products_callback(self, request, response):
        self.get_logger().info(f'Detect products request received for order_id={request.order_id}')

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture frame from camera.')
            response.success = False
            response.message = "Failed to capture frame"
            return response

        # YOLOv8로 객체 인식 수행
        detections = self.detector.detect(frame)
        self.get_logger().info(f'Detected {len(detections)} objects.')

        # 결과 메시지 생성 및 발행
        msg = PickeeVisionDetection()
        msg.robot_id = request.robot_id
        msg.order_id = request.order_id
        msg.success = True
        msg.products = []

        for i, det in enumerate(detections):
            product = PickeeDetectedProduct()
            product.product_id = det['class_id']
            product.bbox_number = i + 1  # bbox_number 채우기
            product.confidence = det['confidence']
            
            detection_info = DetectionInfo()
            
            # Polygon 정보 채우기
            # 임시 Polygon 클래스를 내부에서 정의하여 사용
            class TempPolygon: points = []
            contour = TempPolygon()
            for point in det['polygon']:
                contour.points.append(Point2D(x=float(point[0]), y=float(point[1])))
            detection_info.polygon = contour.points

            # BBox 정보 채우기
            bbox_data = det['bbox']
            detection_info.bbox_coords = BBox(x1=bbox_data[0], y1=bbox_data[1], x2=bbox_data[2], y2=bbox_data[3])

            product.detection_info = detection_info
            msg.products.append(product)

        msg.message = f"{len(msg.products)} products detected."
        self.detection_result_pub.publish(msg)
        self.get_logger().info('Published detection result with polygon and bbox.')

        response.success = True
        response.message = "Detection started"
        return response

    def check_product_in_cart_callback(self, request, response):
        # TODO: 이 로직도 실제 YOLO 추론으로 변경 필요
        self.get_logger().info(f'Check product in cart request: product_id={request.product_id}')
        response.success = True
        response.message = "Cart product check service is not implemented yet."
        return response

    def check_cart_presence_callback(self, request, response):
        # TODO: 이 로직도 실제 YOLO 추론으로 변경 필요
        self.get_logger().info(f'Check cart presence request received.')
        response.success = True
        response.cart_present = True
        response.message = "Cart presence check is not implemented yet."
        return response

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ProductDetectorNode()
        rclpy.spin(node)
    except (IOError, FileNotFoundError) as e:
        print(f"Error starting node: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and node.cap.isOpened():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
