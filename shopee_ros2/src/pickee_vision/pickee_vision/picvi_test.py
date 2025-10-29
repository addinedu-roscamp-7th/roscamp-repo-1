import rclpy
from rclpy.node import Node
import cv2
import os
import numpy as np
from ament_index_python.packages import get_package_share_directory

from collections import Counter

# 분리된 클래스들
from .yolo_detector import YoloDetector
from .cnn_classifier import CnnClassifier
from .udp_video import UdpStreamer

# ROS 관련
from shopee_interfaces.srv import PickeeVisionDetectProducts, PickeeVisionCheckProductInCart, VisionCheckCartPresence, PickeeVisionVideoStreamStart, PickeeVisionVideoStreamStop
from shopee_interfaces.msg import PickeeVisionDetection, DetectedProduct, DetectionInfo, BBox, Point2D, Pose6D, PickeeVisionCartCheck

product_dic = {
    1 : "wasabi", 
    2 : "buldak_can", 
    3 : "butter_can", 
    4 : "richam", 
    5 : "soymilk", 
    6 : "capri_sun", 
    7 : "red_apple", 
    8 : "green_apple", 
    9 : "orange", 
    10 : "pork", 
    11 : "chicken", 
    12 : "fish", 
    13 : "abalone", 
    14 : "eclipse", 
    15 : "ivy", 
    16 : "pepero", 
    17 : "ohyes"
}

class PickeeVisionNode(Node):
    #
    # 모든 장보기 관련 작업을 지휘하는 메인 노드.
    # 실시간 로컬 영상 디스플레이, 서비스 요청 시 인식, 인식 이미지 로컬 및 UDP 스트리밍 기능을 모두 포함합니다.
    #
    def __init__(self):
        super().__init__('picvi_test')

        # --- 의존성 클래스 초기화 (모델 파일 불러오기 위해) ---
        package_share_directory = get_package_share_directory('pickee_vision')
        # 1. 상품 인식용 세그멘테이션 모델
        product_model_path = os.path.join(package_share_directory, '20251027_v11.pt')
        # 2. 장바구니 인식용 클래시피케이션 모델
        cart_model_path = os.path.join(package_share_directory, 'cart_best_.pth')

        try:
            # 1. 상품 인식용 세그멘테이션 모델 로드 (yolo_detector.py 클래스)
            self.product_detector = YoloDetector(product_model_path)
        except FileNotFoundError as e:
            self.get_logger().error(f"YOLO 모델 파일 로드 실패: {e}")
            raise e
        try:
            # 2. 장바구니 인식용 클래시피케이션 모델 로드 (cnn_classifier.py 클래스)
            self.cart_classifier = CnnClassifier(cart_model_path)
        except FileNotFoundError as e:
            self.get_logger().error(f"CNN 모델 파일 로드 실패: {e}")
            raise e

        # ROBOT_ID 설정 (PC가 장착되는 로봇마다 다르게 설정)
        self.ROBOT_ID = 1

        # UDP 스트리머 서버 설정
        self.streamer = UdpStreamer(host='192.168.0.22', port=6000, robot_id=self.ROBOT_ID)
        self.camera_type = ""
        
        # --- 객체 인식 용 로봇팔 웹캠 (arm) ---
        self.arm_cam = cv2.VideoCapture(0)
        if not self.arm_cam.isOpened():
            self.get_logger().error("카메라 인덱스 0을 열 수 없습니다.")
            raise IOError("Cannot open camera 0")
        self.arm_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.arm_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.last_detections = [] # 마지막 인식 결과를 저장하는 변수 -> List

        # --- 장애물 인식 용 카트정면 웹캠 (front) ---
        self.front_cam = cv2.VideoCapture(2)
        if not self.front_cam.isOpened():
            self.get_logger().error("카메라 인덱스 2를 열 수 없습니다.")
            raise IOError("Cannot open camera 2")
        self.front_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.front_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # --- 상품 인식 결과 토픽 ---
        self.detection_result_pub = self.create_publisher(PickeeVisionDetection, '/pickee/vision/detection_result', 10)
        self.cart_check_result_pub = self.create_publisher(PickeeVisionCartCheck, '/pickee/vision/cart_check_result', 10)
        # --- 상품 관련 서비스 ---
        self.create_service(PickeeVisionDetectProducts, '/pickee/vision/detect_products', self.detect_products_callback)
        self.create_service(PickeeVisionCheckProductInCart, '/pickee/vision/check_product_in_cart', self.check_product_in_cart_callback)
        # --- UDP 영상 스트리밍 서비스 ---
        self.create_service(PickeeVisionVideoStreamStart, '/pickee/vision/video_stream_start', self.video_stream_start_callback)
        self.create_service(PickeeVisionVideoStreamStop, '/pickee/vision/video_stream_stop', self.video_stream_stop_callback)
        # --- 장바구니 유무 확인 서비스 ---
        self.create_service(VisionCheckCartPresence, '/pickee/vision/check_cart_presence', self.check_cart_presence_callback)

        # 메인 루프 타이머 (60 FPS)
        self.main_loop_timer = self.create_timer(1.0 / 60.0, self.main_loop)
        self.get_logger().info('Product Detector Node has been started.')

    def main_loop(self):
        # 상시 실행되는 메인 루프: 영상처리, 로컬 디스플레이, UDP 큐잉 담당
        ret_arm, arm_frame = self.arm_cam.read()
        ret_front, front_frame = self.front_cam.read()
        
        if not ret_arm and not ret_front: # 둘 다 실패하면 리턴
            self.get_logger().warn('Failed to capture frame from both cameras.')
            return
        
        # 로컬 디스플레이
        annotated_frame = self.draw_annotations(arm_frame.copy(), self.last_detections)
        cv2.imshow("Detection Result", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('"q" pressed, shutting down node.')
            self.destroy_node()
            if rclpy.ok(): rclpy.get_current_context().shutdown()
            
        # UDP 스트리밍 처리 - 테스트용
        if self.streamer.is_running:
            if self.camera_type == "arm" and ret_arm:
                self.streamer.send_frame(arm_frame)
            elif self.camera_type == "front" and ret_front:
                self.streamer.send_frame(front_frame)
            else:
                self.get_logger().warn(f"Streaming requested for {self.camera_type} but frame not available or type invalid.")

    def draw_annotations(self, frame, detections):
        # 주어진 프레임에 감지된 객체 정보를 그립니다.
        for i, det in enumerate(detections):
            bbox_data = det['bbox']
            cv2.rectangle(frame, (bbox_data[0], bbox_data[1]), (bbox_data[2], bbox_data[3]), (0, 255, 0), 2)
            # polygon_pts = np.array(det['polygon'], np.int32)
            # cv2.polylines(frame, [polygon_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, f"# {i + 1}: {product_dic[det['class_id']]}", (bbox_data[0], bbox_data[1] - 15), 
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
        return frame
    
    def detect_products_callback(self, request, response):
        # 서비스 요청 시 1회 인식 및 데이터 발행
        self.get_logger().info(f'Detect products request received for order_id={request.order_id}')
        ret, frame = self.arm_cam.read()
        if not ret:
            self.get_logger().error('Failed to capture frame for detection.')
            response.success = False
            response.message = "Failed to capture frame"
            return response

        # 인식 수행 및 결과 저장 (yolo_detector.py 클래스)
        self.last_detections = self.product_detector.detect(frame)
        self.get_logger().info(f'Detected {len(self.last_detections)} objects.')

        # 요청된 상품 ID의 필요 수량과 실제 감지된 수량을 비교 변수 설정
        requested_counts = Counter(request.product_ids)
        detected_ids_list = [det['class_id'] for det in self.last_detections]
        detected_counts = Counter(detected_ids_list)
        missing_products = {} # 개수 맞지 않는 상품 딕셔너리 {상품ID:개수}
        all_products_found = True
        # 요청된 상품 ID의 필요 수량과 실제 감지된 수량을 비교
        for product_id, required_count in requested_counts.items():
            detected_count = detected_counts.get(product_id, 0)
            if detected_count < required_count:
                all_products_found = False
                missing_products[product_id] = required_count - detected_count
        
        if all_products_found:
            # 모든 요청된 상품이 감지되었을 경우
            response.success = True
            response.message = "All requested products and quantities have been detected."
            # self.get_logger().info('All requested products and quantities found.')
            
            # 인식 데이터를 ROS 토픽으로 발행
            self.publish_detection_data(request.robot_id, request.order_id)
        else:
            # 요청된 상품 중 일부 또는 전체가 감지되지 않았을 경우
            response.success = False
            message = "Failed to detect all requested products. Missing quantities: "
            missing_str = ", ".join([f"Product ID {pid}: {qty} missing" for pid, qty in missing_products.items()])
            response.message = message + missing_str
            # self.get_logger().warning(f'Missing products/quantities: {missing_str}')
        
        return response

    def publish_detection_data(self, robot_id, order_id):
        # self.last_detections를 ROS 메시지로 변환하여 발행
        detected_products = [] # DetectedProduct[] 자료형
        for i, det in enumerate(self.last_detections):
            contour_points = [Point2D(x=float(p[0]), y=float(p[1])) for p in det['polygon']]
            bbox_data = det['bbox']
            bbox_msg = BBox(x1=bbox_data[0], y1=bbox_data[1], x2=bbox_data[2], y2=bbox_data[3])
            detection_info_msg = DetectionInfo(polygon=contour_points, bbox_coords=bbox_msg)
            pose6d_msg = Pose6D()
            product = DetectedProduct(
                product_id=det['class_id'],
                confidence=det['confidence'],
                bbox=bbox_msg,
                bbox_number=i + 1,
                detection_info=detection_info_msg,
                pose=pose6d_msg
            )
            detected_products.append(product)
        
        msg = PickeeVisionDetection(
            robot_id=robot_id,
            order_id=order_id,
            success=True,
            products=detected_products,
            message=f"{len(detected_products)} products detected."
        )
        self.detection_result_pub.publish(msg)
        # self.get_logger().info(f'Published {len(detected_products)} detection data to topic.')

    def video_stream_start_callback(self, request, response):
        self.get_logger().info(f'Video stream start service called for camera: {request.camera_type}.')
        self.camera_type = request.camera_type # camera_type 반영
        self.streamer.start()
        response.success = True
        response.message = "UDP streamer started."
        return response

    def video_stream_stop_callback(self, request, response):
        self.get_logger().info('Video stream stop service called.')
        self.streamer.stop()
        self.camera_type = "" # camera_type 리셋
        response.success = True
        response.message = "UDP streamer stopped."
        return response

    def check_product_in_cart_callback(self, request, response):
        self.get_logger().info(f'Check product in cart request received for product_id: {request.product_id}')
        
        ret, frame = self.arm_cam.read() # 로봇팔 카메라 사용
        if not ret:
            self.get_logger().error('Failed to capture frame for product in cart check.')
            # 실패 메시지 발행
            result_msg = PickeeVisionCartCheck(
                robot_id=request.robot_id,
                order_id=request.order_id,
                success=False,
                product_id=request.product_id,
                found=False,
                quantity=0,
                message='Failed to capture frame'
            )
            self.cart_check_result_pub.publish(result_msg)
            
            # 서비스 응답
            response.success = False
            response.message = 'Failed to capture frame'
            return response

        # 2. 프레임에서 상품 감지
        self.last_detections = self.product_detector.detect(frame)
        
        # 3. 요청된 상품 개수 세기
        requested_product_id = request.product_id
        quantity = 0
        for det in self.last_detections:
            if det['class_id'] == requested_product_id:
                quantity += 1
        
        found = quantity > 0
        self.get_logger().info(f'Found {quantity} instances of product_id {requested_product_id}.')

        # 4. 토픽으로 결과 발행
        result_msg = PickeeVisionCartCheck(
            robot_id=request.robot_id,
            order_id=request.order_id,
            success=True, # 감지 프로세스는 성공
            product_id=requested_product_id,
            found=found,
            quantity=quantity,
            message=f'Found {quantity} of product {requested_product_id}'
        )
        self.cart_check_result_pub.publish(result_msg)
        # self.get_logger().info('Published cart check result.')

        # 5. 서비스로 응답
        response.success = found
        response.message = f'Detection complete. product_id: {requested_product_id}, Quantity: {quantity}'
        
        return response
    
    def check_cart_presence_callback(self, request, response):
        # 서비스 요청 시 장바구니 존재 여부를 클래시피케이션 모델로 확인
        self.get_logger().info('Check cart presence request received.')
        ret, frame = self.arm_cam.read() # 로봇팔 카메라 사용
        if not ret:
            self.get_logger().error('Failed to capture frame for cart presence check.')
            response.success = False
            response.cart_present = False
            response.confidence = 0
            response.message = "Failed to capture frame"
            return response

        # 이전 상품 인식 정보 화면에서 지우기
        self.last_detections = []

        # CNN 분류 수행
        class_id, confidence, class_name = self.cart_classifier.classify(frame)

        # 'empty_cart'는 장바구니 존재, 'full_cart'는 장바구니 없음(사용 불가)으로 판단
        if class_name == 'empty_cart' and confidence >= 90:
            self.get_logger().info(f'Empty cart detected with confidence: {confidence:.2f}')
            response.success = True
            response.cart_present = True
            response.confidence = confidence
            response.message = f'Empty cart is present (confidence: {confidence:.2f})'
        elif class_name == 'full_cart':
            self.get_logger().info(f'Full cart detected with confidence: {confidence:.2f}. Considering as not present for pickup.')
            response.success = False
            response.cart_present = False
            response.confidence = confidence
            response.message = f'Cart is full, not available for use (confidence: {confidence:.2f})'
        elif class_name == 'no_cart':
            self.get_logger().info(f'NO cart detected with confidence: {confidence:.2f}. Considering as not present for pickup.')
            response.success = False
            response.cart_present = False
            response.confidence = confidence
            response.message = f'Cart isn\'t here, not available for use (confidence: {confidence:.2f})'
        elif class_name == 'error':
            self.get_logger().error('An error occurred during cart classification.')
            response.success = False
            response.cart_present = False
            response.confidence = confidence
            response.message = 'An error occurred during classification.'
        else:
            # 이 경우는 모델이 'empty_cart'나 'full_cart'나 'no_cart'가 아닌 다른 것을 예측한 경우
            self.get_logger().info(f'Cart not detected. Classified as: {class_name}')
            response.success = False
            response.cart_present = False
            response.confidence = confidence
            response.message = f'Cart is not present. (classified as: {class_name})'
            
        return response

    def destroy_node(self):
        self.get_logger().info("Shutting down node.")
        self.streamer.stop()
        if self.arm_cam.isOpened(): self.arm_cam.release()
        if self.front_cam.isOpened(): self.front_cam.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PickeeVisionNode()
        rclpy.spin(node)
    except (IOError, FileNotFoundError) as e:
        print(f"Error starting node: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        if node: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()