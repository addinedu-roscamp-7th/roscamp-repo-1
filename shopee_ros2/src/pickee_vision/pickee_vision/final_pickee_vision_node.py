import rclpy
from rclpy.node import Node
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from collections import Counter
import numpy as np
import torch
from torchvision import models, transforms
import time
import collections

# --- 분리된 클래스들 ---
from .yolo_detector import YoloDetector
from .cnn_classifier import CnnClassifier
from .udp_video import UdpStreamer
from .pose_cnn_model import PoseCNN

# --- ROS 인터페이스 ---
# 서비스
from shopee_interfaces.srv import (
    PickeeVisionDetectProducts, 
    PickeeVisionCheckProductInCart, 
    VisionCheckCartPresence, 
    PickeeVisionVideoStreamStart, 
    PickeeVisionVideoStreamStop,
    ArmPickProduct,
)
from std_srvs.srv import Trigger
# 메시지
from shopee_interfaces.msg import (
    PickeeVisionDetection, 
    DetectedProduct, 
    DetectionInfo, 
    BBox, 
    Point2D, 
    Pose6D, 
    PickeeVisionCartCheck,
)
from std_msgs.msg import Bool

product_dic = {
    1 : "wasabi", 2 : "buldak_can", 3 : "butter_can", 4 : "richam", 5 : "soymilk", 
    6 : "capri_sun", 7 : "red_apple", 8 : "green_apple", 9 : "orange", 10 : "pork", 
    11 : "chicken", 12 : "fish", 13 : "abalone", 14 : "eclipse", 15 : "ivy", 
    16 : "pepero", 17 : "ohyes"
}

class FinalPickeeVisionNode(Node):
    def __init__(self):
        super().__init__('final_pickee_vision_node')
        self.get_logger().info("Initializing Final Pickee Vision Node...")

        # =================================================================
        # 1. pickee_vision_node.py 기반 초기화
        # =================================================================
        package_share_directory = get_package_share_directory('pickee_vision').replace(
                            'install/pickee_vision/share/pickee_vision', 
                            'src/pickee_vision/resource'
                        )
        
        # --- 모델 경로 설정 ---
        product_model_path = os.path.join(package_share_directory, '20251104_v11_ver1_ioudefault.pt')
        cart_model_path = os.path.join(package_share_directory, 'cart_best_.pth')
        
        # --- 모델 로드 (YOLO, Cart CNN) ---
        try:
            self.product_detector = YoloDetector(product_model_path)
            self.cart_classifier = CnnClassifier(cart_model_path)
            self.get_logger().info("YOLO detector and Cart Classifier loaded.")
        except FileNotFoundError as e:
            self.get_logger().error(f"Failed to load base models: {e}")
            raise e

        self.ROBOT_ID = 1
        self.streamer = UdpStreamer(host='192.168.0.154', port=6000, robot_id=self.ROBOT_ID)
        self.camera_type = ""
        
        # --- 카메라 초기화 ---
        self.arm_cam = cv2.VideoCapture(4)
        if not self.arm_cam.isOpened():
            self.get_logger().error("Cannot open camera index 4 (arm_cam).")
            raise IOError("Cannot open camera 4")
        self.arm_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.arm_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.front_cam = cv2.VideoCapture(6)
        if not self.front_cam.isOpened():
            self.get_logger().error("Cannot open camera index 6 (front_cam).")
            raise IOError("Cannot open camera 6")
        self.front_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.front_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.last_detections = []
        self.results = None

        # --- 기존 서비스 및 토픽 ---
        self.detection_result_pub = self.create_publisher(PickeeVisionDetection, '/pickee/vision/detection_result', 10)
        self.cart_check_result_pub = self.create_publisher(PickeeVisionCartCheck, '/pickee/vision/cart_check_result', 10)
        self.create_service(PickeeVisionDetectProducts, '/pickee/vision/detect_products', self.detect_products_callback)
        self.create_service(PickeeVisionCheckProductInCart, '/pickee/vision/check_product_in_cart', self.check_product_in_cart_callback)
        self.create_service(PickeeVisionVideoStreamStart, '/pickee/vision/video_stream_start', self.video_stream_start_callback)
        self.create_service(PickeeVisionVideoStreamStop, '/pickee/vision/video_stream_stop', self.video_stream_stop_callback)
        self.create_service(VisionCheckCartPresence, '/pickee/vision/check_cart_presence', self.check_cart_presence_callback)

        # =================================================================
        # 2. pose_predictor_test_node.py 기반 초기화
        # =================================================================
        
        # --- 상태 및 멤버 변수 ---
        self.state = 'IDLE'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.real_cur_cord = None
        self.cnn_servoing_timer = None
        
        # --- PID 제어기 파라미터 및 변수 ---
        self.KP = 0.4   # P 제어기 게인
        self.KI = 0.007 # I 제어기 게인
        self.KD = 0.05  # D 제어기 게인
        self.CONVERGENCE_THRESHOLD = 5
        self.integral_error = np.zeros(6, dtype=np.float32) # I 제어를 위한 이전 에러
        self.previous_error = np.zeros(6, dtype=np.float32) # D 제어를 위한 이전 에러
        self.last_servoing_time = None
        self.integral_clamp = 2.0 

        # --- 모델 및 리소스 경로 ---
        cnn_model_path = os.path.join(package_share_directory, "20251112_total.pt")
        # target_image_path = os.path.join(package_share_directory, "test/capture_20251107-174001.jpg")
        self.target_image_path_fish = os.path.join(package_share_directory, "test/target_fish_5.jpg")
        self.target_image_path_eclipse = os.path.join(package_share_directory, "test/target_eclipse_3.jpg")

        # --- 역정규화 파라미터 ---
        self.pose_mean = np.array([-75.24822998046875, 140.6298370361328, 220.1119842529297, -179.412109375, 0.4675877094268799, 44.999176025390625], dtype=np.float32)
        self.pose_std = np.array([31.43274688720703, 30.908634185791016, 1.6736514568328857, 0.38159045577049255, 1.5441224575042725, 25.34377098083496], dtype=np.float32)

        # --- PoseCNN 모델 로드 ---
        try:
            self.pose_cnn_model = PoseCNN(num_classes=2).to(self.device)
            self.pose_cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
            self.pose_cnn_model.eval()
            self.get_logger().info("PoseCNN model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load PoseCNN model: {e}")
            raise e

        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])

        # --- 목표 이미지 Pose 추론 ---
        # try:
        #     target_image = cv2.imread(target_image_path)
        #     norm_tar_cord = self.predict_pose(target_image)
        #     self.tar_cord = self.de_standardize_pose(norm_tar_cord)
        #     self.get_logger().info(f"Target pose predicted and denormalized: {self.tar_cord}")
        # except Exception as e:
        #     self.get_logger().error(f"Failed to predict target pose: {e}")
        #     raise e
        self.tar_cord = None

        # --- Arm 연동 ROS2 인터페이스 ---
        self.start_pick_srv = self.create_service(ArmPickProduct, '/pickee/arm/pick_product_gg', self.start_pick_sequence_callback)
        self.move_start_client = self.create_client(Trigger, '/pickee/arm/move_start')
        self.arm_ready_sub = self.create_subscription(Bool, '/pickee/arm/is_moving', self.arm_ready_callback, 10)
        self.pose_subscriber = self.create_subscription(Pose6D, '/pickee/arm/real_pose', self.real_pose_callback, 10)
        self.move_publisher = self.create_publisher(Pose6D, '/pickee/arm/move_servo', 10)
        self.grep_product_client = self.create_client(Trigger, '/pickee/arm/grep_product')

        # =================================================================
        # 3. 메인 루프 시작
        # =================================================================
        self.main_loop_timer = self.create_timer(1.0 / 60.0, self.main_loop)
        self.get_logger().info('Final Pickee Vision Node has been started.')

    # =====================================================================
    # 기존 pickee_vision_node.py 메소드들 (main_loop, draw_annotations 등)
    # =====================================================================
    def main_loop(self):
        ret_arm, arm_frame = self.arm_cam.read()
        ret_front, front_frame = self.front_cam.read()
        
        if not ret_arm and not ret_front:
            self.get_logger().warn('Failed to capture frame from both cameras.')
            return
        
        # 로컬 디스플레이: 상태에 따라 다른 프레임 표시
        if self.state == 'IDLE' or self.state == 'SHELF_VIEW_READY' or self.state == 'WAITING_FOR_SHELF_VIEW':
            annotated_frame = self.draw_annotations(arm_frame.copy(), self.last_detections)
            cv2.imshow("Detection Result", annotated_frame)
        elif self.state == 'CNN_SERVOING':
            # cnn_servoing_loop에서 자체적으로 imshow 호출
            pass
        else: # 다른 상태일 때 기본 화면
            cv2.imshow("Detection Result", arm_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('"q" pressed, shutting down node.')
            self.destroy_node()
            if rclpy.ok(): rclpy.get_current_context().shutdown()

        if self.streamer.is_running:
            if self.camera_type == "arm" and ret_arm: self.streamer.send_frame(arm_frame)
            elif self.camera_type == "front" and ret_front: self.streamer.send_frame(front_frame)

    def draw_annotations(self, frame, detections):
        if self.results is not None:
            frame = self.results[0].plot()

        for i, det in enumerate(detections):
            bbox_data = det['bbox']
            cv2.rectangle(frame, (bbox_data[0], bbox_data[1]), (bbox_data[2], bbox_data[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"# {i + 1}: {product_dic[int(det['class_name'])]}", (bbox_data[0], bbox_data[1] - 25), 
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            # cv2.putText(frame, f"# {i + 1}: {product_dic.get(det['class_id'], 'Unknown')}", (bbox_data[0], bbox_data[1] - 15), 
            #             cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
        return frame
    
    def detect_products_callback(self, request, response):
        # 상태가 SHELF_VIEW_READY가 아니면, 기존의 독립적인 감지 기능으로만 작동
        if self.state != 'SHELF_VIEW_READY':
            self.get_logger().info(f'Detect products request received (Standalone Mode). Current state: {self.state}')
            # IDLE 상태가 아닌 다른 피킹 상태에서는 이 요청을 받으면 안됨
            if self.state != 'IDLE':
                response.success = False
                response.message = f"Cannot process standalone detection in state '{self.state}'."
                self.get_logger().warn(response.message)
                return response
        else:
             self.get_logger().info(f'Detect products request received (Picking Sequence Mode).')

        # --- 공통 감지 로직 ---
        ret_arm, frame_arm = self.arm_cam.read()
        if not ret_arm:
            self.get_logger().error('Failed to capture frame for detection.')
            response.success = False
            response.message = "Failed to capture frame"
            if self.state == 'SHELF_VIEW_READY': self.set_state('IDLE') # 시퀀스 중 카메라 실패 시에는 IDLE로 리셋
            return response

        self.last_detections, self.results = self.product_detector.detect(frame_arm)
        self.last_detections.sort(key=lambda det: (det['bbox'][0], det['bbox'][1]))
        self.get_logger().info(f'Detected {len(self.last_detections)} objects.')

        requested_counts = Counter(request.product_ids)
        detected_ids_list = [int(det['class_name']) for det in self.last_detections]
        detected_counts = Counter(detected_ids_list)
        missing_products = {}
        all_products_found = True
        for product_id, required_count in requested_counts.items():
            detected_count = detected_counts.get(product_id, 0)
            if detected_count < required_count:
                all_products_found = False
                missing_products[product_id] = required_count - detected_count
        
        # --- 결과에 따른 분기 ---
        if all_products_found:
            response.success = True
            response.message = "All requested products and quantities have been detected."
            self.publish_detection_data(request.robot_id, request.order_id)
            # 피킹 시퀀스 중이었다면 다음 상태로 전환
            if self.state == 'SHELF_VIEW_READY':
                self.get_logger().info("Proceeding to next step in picking sequence.")
                self.set_state('WAITING_FOR_TOP_VIEW')
        else:
            response.success = False
            missing_str = ", ".join([f"Product ID {pid}: {qty} missing" for pid, qty in missing_products.items()])
            response.message = "Failed to detect all requested products. Missing quantities: " + missing_str
            # 피킹 시퀀스 중 감지 실패 시, 상태를 유지하고 재요청을 기다림
            if self.state == 'SHELF_VIEW_READY':
                self.get_logger().warn("Detection failed, remaining in SHELF_VIEW_READY for retry.")
        
        return response

    def publish_detection_data(self, robot_id, order_id):
        self.last_detections.sort(key=lambda det: (det['bbox'][0], det['bbox'][1]))
        detected_products = []
        for i, det in enumerate(self.last_detections):
            contour_points = [Point2D(x=float(p[0]), y=float(p[1])) for p in det['polygon']]
            bbox_msg = BBox(x1=det['bbox'][0], y1=det['bbox'][1], x2=det['bbox'][2], y2=det['bbox'][3])
            product = DetectedProduct(
                product_id=int(det['class_name']), confidence=det['confidence'], bbox=bbox_msg, bbox_number=i + 1,
                detection_info=DetectionInfo(polygon=contour_points, bbox_coords=bbox_msg),
                pose=Pose6D()
            )
            detected_products.append(product)
        
        msg = PickeeVisionDetection(
            robot_id=robot_id, order_id=order_id, success=True,
            products=detected_products, message=f"{len(detected_products)} products detected."
        )
        self.detection_result_pub.publish(msg)

    def video_stream_start_callback(self, request, response):
        self.get_logger().info(f'Video stream start for camera: {request.camera_type}.')
        self.camera_type = request.camera_type
        self.streamer.start()
        response.success = True
        response.message = "UDP streamer started."
        return response

    def video_stream_stop_callback(self, request, response):
        self.get_logger().info('Video stream stop called.')
        self.streamer.stop()
        self.camera_type = ""
        response.success = True
        response.message = "UDP streamer stopped."
        return response

    def check_product_in_cart_callback(self, request, response):
        self.get_logger().info(f'Check product in cart for product_id: {request.product_id}')
        ret_arm, frame_arm = self.arm_cam.read()
        if not ret_arm:
            response.success = False; response.message = 'Failed to capture frame'; return response

        self.last_detections = self.product_detector.detect(frame_arm)
        quantity = sum(1 for det in self.last_detections if int(det['class_name']) == request.product_id)
        found = quantity > 0
        
        result_msg = PickeeVisionCartCheck(
            robot_id=request.robot_id, order_id=request.order_id, success=True,
            product_id=request.product_id, found=found, quantity=quantity,
            message=f'Found {quantity} of product {request.product_id}'
        )
        self.cart_check_result_pub.publish(result_msg)
        
        response.success = found
        response.message = f'Detection complete. Quantity: {quantity}'
        return response
    
    def check_cart_presence_callback(self, request, response):
        self.get_logger().info('Check cart presence request received.')
        ret_arm, frame_arm = self.arm_cam.read()
        if not ret_arm:
            response.success=False; response.cart_present=False; response.confidence=0.0; response.message="Failed to capture frame"; return response

        self.last_detections = []
        class_id, confidence, class_name = self.cart_classifier.classify(frame_arm)

        if class_name == 'empty_cart' and confidence >= 90:
            response.success=True; response.cart_present=True; response.message='Empty cart is present'
        else:
            response.success=False; response.cart_present=False; response.message=f'Cart not available (classified as: {class_name})'
        response.confidence = confidence
        return response

    # =====================================================================
    # pose_predictor_test_node.py 메소드들 (상태 머신, 서보잉 등)
    # =====================================================================
    def set_state(self, new_state):
        self.get_logger().info(f"State transition: {self.state} -> {new_state}")
        self.state = new_state

    def start_pick_sequence_callback(self, request, response):
        if self.state != 'IDLE':
            response.success = False
            response.message = f"Node is not in IDLE state, current state: {self.state}"
            return response

        # --- Target Image 및 Pose 설정 ---
        product_id = request.product_id
        self.get_logger().info(f"Start pick sequence for product_id: {product_id}")

        if product_id == 12: # fish
            target_image_path = self.target_image_path_fish
        elif product_id == 14: # eclipse
            target_image_path = self.target_image_path_eclipse
        else:
            response.success = False
            response.message = f"Unsupported product_id for picking: {product_id}"
            self.get_logger().error(response.message)
            return response

        try:
            target_image = cv2.imread(target_image_path)
            if target_image is None:
                raise FileNotFoundError(f"Target image not found at {target_image_path}")
            norm_tar_cord = self.predict_pose(target_image)
            self.tar_cord = self.de_standardize_pose(norm_tar_cord)
            self.get_logger().info(f"Target pose for product {product_id} calculated: {self.tar_cord}")
        except Exception as e:
            self.get_logger().error(f"Failed to calculate target pose: {e}")
            response.success = False
            response.message = f"Failed to calculate target pose: {e}"
            return response

        # --- 로봇 팔 이동 요청 ---
        if not self.move_start_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Arm move_start service not available.')
            response.success = False; response.message = "Arm move_start service not available."; return response

        # self.get_logger().info(f"Start pick sequence trigger received. Moving arm to shelf view.")
        self.set_state('WAITING_FOR_SHELF_VIEW')

        arm_request = Trigger.Request()
        future = self.move_start_client.call_async(arm_request)
        future.add_done_callback(self.move_start_response_callback)

        response.success = True
        response.message = "Request accepted. Moving arm to shelf_view."
        return response

    def move_start_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Arm reached shelf_view. Ready for detection command.")
                self.set_state('SHELF_VIEW_READY')
            else:
                self.get_logger().error(f"Arm failed to start move_start: {response.message}")
                self.set_state('IDLE')
        except Exception as e:
            self.get_logger().error(f"Service call to move_start failed: {e}")
            self.set_state('IDLE')

    def arm_ready_callback(self, msg):
        if msg.data is True and self.state == 'WAITING_FOR_TOP_VIEW':
            self.get_logger().info("Arm is ready for CNN servoing. Starting...")
            self.integral_error = np.zeros(6, dtype=np.float32)
            self.previous_error = np.zeros(6, dtype=np.float32)
            self.last_servoing_time = None
            self.set_state('CNN_SERVOING')
            if self.cnn_servoing_timer is None or self.cnn_servoing_timer.is_canceled():
                self.cnn_servoing_timer = self.create_timer(0.03, self.cnn_servoing_loop)
    
    def real_pose_callback(self, msg):
        self.real_cur_cord = np.array([msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz])

    def cnn_servoing_loop(self):
        if self.state != 'CNN_SERVOING': return
        ret, frame = self.arm_cam.read()
        if not ret or self.real_cur_cord is None: return
        # frame = self.arm_cam.read()[1] ###########추가###########
        # if frame is None or self.real_cur_cord is None: return ###########추가###########

        current_time = self.get_clock().now()
        if self.last_servoing_time is None:
            self.last_servoing_time = current_time
            return
        dt = (current_time - self.last_servoing_time).nanoseconds / 1e9
        self.last_servoing_time = current_time

        norm_cur_cord = self.predict_pose(frame)
        cur_cord = self.de_standardize_pose(norm_cur_cord)
        pose_err = cur_cord - self.real_cur_cord
        real_tar_cord = self.tar_cord - pose_err
        error = real_tar_cord - self.real_cur_cord
        error_magnitude = np.linalg.norm(np.array([error[0], error[1], error[5]]))
        self.get_logger().info(f"CNN Visual Servoing... Error: {error_magnitude:.3f}")

        cv2.imshow("CNN Servoing", frame)
        cv2.waitKey(1)

        if error_magnitude < self.CONVERGENCE_THRESHOLD:
            self.get_logger().info(f"Target reached. Calling grep_product service.")
            if self.cnn_servoing_timer: self.cnn_servoing_timer.cancel()
            
            if not self.grep_product_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error('Grep product service not available.')
                self.set_state('IDLE'); return

            request = Trigger.Request()
            future = self.grep_product_client.call_async(request)
            future.add_done_callback(self.grep_product_response_callback)
            self.set_state('IDLE')
        else:
            # --- PID 제어 계산 ---
            # Proportional (P)
            proportional_term = self.KP * error

            # Integral (I)
            self.integral_error += error * dt
            np.clip(self.integral_error, -self.integral_clamp, self.integral_clamp, out=self.integral_error)
            integral_term = self.KI * self.integral_error

            # Derivative (D)
            derivative_error = (error - self.previous_error) / dt
            derivative_term = self.KD * derivative_error
            self.previous_error = error

            # PI 제어 (X, Y)
            PI_delta = proportional_term + integral_term
            commanded_pose = self.real_cur_cord + PI_delta
            
            # PD 제어 (RZ - Yaw)
            commanded_pose[5] = self.real_cur_cord[5] + proportional_term[5] + derivative_term[5]
            
            move_cmd = Pose6D()
            
            move_cmd.x, move_cmd.y, move_cmd.rz = float(commanded_pose[0]), float(commanded_pose[1]), float(commanded_pose[5])
            move_cmd.rx, move_cmd.ry, move_cmd.z = float(self.tar_cord[3]), float(self.tar_cord[4]), float(self.tar_cord[2])
            
            self.move_publisher.publish(move_cmd)
            time.sleep(2.5)

    def grep_product_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Grep product service call successful: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Grep service call failed: {e}")

    def predict_pose(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pose_pred, _ = self.pose_cnn_model(img_tensor)
            return pose_pred.cpu().numpy().flatten()

    def de_standardize_pose(self, norm_pose):
        return (norm_pose * self.pose_std) + self.pose_mean

    def destroy_node(self):
        self.get_logger().info("Shutting down node.")
        self.streamer.stop()
        if self.arm_cam.isOpened(): self.arm_cam.release()
        if self.front_cam.isOpened(): self.front_cam.release()
        cv2.destroyAllWindows()
        if self.cnn_servoing_timer: self.cnn_servoing_timer.cancel()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FinalPickeeVisionNode()
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
