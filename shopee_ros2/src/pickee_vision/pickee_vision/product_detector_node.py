import rclpy
from rclpy.node import Node
import cv2
import os
import numpy as np
from ament_index_python.packages import get_package_share_directory
import queue

# 분리된 클래스들
from .yolo_detector import YoloDetector
from .udp_video import UdpStreamer

# ROS 관련
from shopee_interfaces.srv import PickeeVisionDetectProducts, PickeeVisionVideoStreamStart, PickeeVisionVideoStreamStop
from shopee_interfaces.msg import PickeeVisionDetection, DetectedProduct, BBox, Point2D, DetectionInfo

class ProductDetectorNode(Node):
    # 모든 Vision 관련 작업을 지휘하는 메인 노드.
    def __init__(self):
        super().__init__('product_detector_node')

        # --- 의존성 클래스 초기화 ---
        # YOLO 감지기
        package_share_directory = get_package_share_directory('pickee_vision')
        model_path = os.path.join(package_share_directory, '20251015_1.pt')
        try:
            self.detector = YoloDetector(model_path)
        except FileNotFoundError as e:
            self.get_logger().error(f"YOLO 모델 파일 로드 실패: {e}")
            raise e

        # UDP 스트리머
        self.streamer = UdpStreamer(host='127.0.0.1', port=6000, robot_id=1)

        # --- 하드웨어 및 상태 초기화 ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("카메라 인덱스 1을 열 수 없습니다.")
            raise IOError("Cannot open camera 1")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.last_detections = []

        # --- ROS2 인터페이스 생성 ---
        self.detection_result_pub = self.create_publisher(PickeeVisionDetection, '/pickee/vision/detection_result', 10)
        self.create_service(PickeeVisionDetectProducts, '/pickee/vision/detect_products', self.detect_products_callback)
        self.create_service(PickeeVisionVideoStreamStart, '/pickee/vision/video_stream_start', self.video_stream_start_callback)
        self.create_service(PickeeVisionVideoStreamStop, '/pickee/vision/video_stream_stop', self.video_stream_stop_callback)

        # 메인 루프 타이머 (30 FPS)
        self.main_loop_timer = self.create_timer(1.0 / 30.0, self.main_loop)
        self.get_logger().info('Product Detector Node (Refactored) has been started.')

    def main_loop(self):
        # 상시 실행되는 메인 루프: 영상처리, 로컬 디스플레이, UDP 큐잉 담당
        ret, frame = self.cap.read()
        if not ret:
            return

        # 항상 최신 인식 결과를 화면에 그림
        annotated_frame = self.draw_annotations(frame.copy(), self.last_detections)

        # 로컬 화면에 표시
        cv2.imshow("Detection Result", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('"q" pressed, shutting down node.')
            self.destroy_node()
            if rclpy.ok(): rclpy.get_current_context().shutdown()

        # 스트리밍이 켜져 있으면, 완성된 프레임을 스트리머의 큐에 넣음
        if self.streamer.is_running:
            self.streamer.queue_frame(annotated_frame)

    def detect_products_callback(self, request, response):
        """서비스 요청 시 1회 인식 및 데이터 발행"""
        self.get_logger().info(f'Detect products request received for order_id={request.order_id}')
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture frame for detection.')
            response.success = False
            response.message = "Failed to capture frame"
            return response

        # 인식 수행 및 결과 저장
        self.last_detections = self.detector.detect(frame)
        self.get_logger().info(f'Detected {len(self.last_detections)} objects.')

        # 인식 데이터를 ROS 메시지로 변환하여 발행
        self.publish_detection_data(request.robot_id, request.order_id)

        response.success = True
        response.message = "Detection successful and data published."
        return response

    def publish_detection_data(self, robot_id, order_id):
        """self.last_detections를 ROS 메시지로 변환하여 발행"""
        detected_products = []
        for i, det in enumerate(self.last_detections):
            contour_points = [Point2D(x=float(p[0]), y=float(p[1])) for p in det['polygon']]
            bbox_data = det['bbox']
            bbox_msg = BBox(x1=bbox_data[0], y1=bbox_data[1], x2=bbox_data[2], y2=bbox_data[3])
            detection_info_msg = DetectionInfo(polygon=contour_points, bbox_coords=bbox_msg)
            product = DetectedProduct(
                product_id=det['class_name'],
                bbox_number=i + 1,
                confidence=det['confidence'],
                detection_info=detection_info_msg
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
        self.get_logger().info(f'Published {len(detected_products)} detection data to topic.')

    def draw_annotations(self, frame, detections):
        """주어진 프레임에 감지된 객체 정보를 그립니다."""
        for i, det in enumerate(detections):
            bbox_data = det['bbox']
            cv2.rectangle(frame, (bbox_data[0], bbox_data[1]), (bbox_data[2], bbox_data[3]), (0, 255, 0), 2)
            polygon_pts = np.array(det['polygon'], np.int32)
            cv2.polylines(frame, [polygon_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, f"# {i + 1}", (bbox_data[0], bbox_data[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return frame

    def video_stream_start_callback(self, request, response):
        self.get_logger().info('Video stream start service called.')
        self.streamer.start()
        response.success = True
        response.message = "UDP streamer started."
        return response

    def video_stream_stop_callback(self, request, response):
        self.get_logger().info('Video stream stop service called.')
        self.streamer.stop()
        response.success = True
        response.message = "UDP streamer stopped."
        return response

    def destroy_node(self):
        self.get_logger().info("Shutting down node.")
        self.streamer.stop()
        if self.cap.isOpened(): self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ProductDetectorNode()
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
