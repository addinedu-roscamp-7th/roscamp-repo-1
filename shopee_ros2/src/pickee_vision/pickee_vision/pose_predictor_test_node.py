import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import Pose6D
from shopee_interfaces.srv import ArmPickProduct
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

import cv2
import threading
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import time
import socket
import collections
from ultralytics import YOLO

# =================================================================
# 1. 비디오 수신을 위한 VideoReceiver 클래스 (기존과 동일)
# =================================================================
class VideoReceiver:
    def __init__(self, port=6230):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.running = True
        self.frame_queue = collections.deque(maxlen=1)  # 버퍼 크기가 1인 큐를 사용하여 최신 프레임만 저장
        self.packet_buffer = {}

    def run(self):
        try:
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(65536)
                    if b'||' not in data: continue
                    header, img_data = data.split(b'||', 1)
                    frame_id, packet_num, total_packets = map(int, header.decode().split(','))
                    if frame_id not in self.packet_buffer: self.packet_buffer[frame_id] = [None] * total_packets
                    self.packet_buffer[frame_id][packet_num] = img_data
                    if all(p is not None for p in self.packet_buffer[frame_id]):
                        complete_data = b''.join(self.packet_buffer[frame_id])
                        np_data = np.frombuffer(complete_data, np.uint8)
                        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.frame_queue.append(frame)  # 새 프레임을 큐에 추가 (오래된 프레임은 자동 삭제)
                        del self.packet_buffer[frame_id]
                except Exception as e:
                    print(f"[VideoReceiver] Error: {e}")
                    continue
        finally:
            self.sock.close()

    def get_frame(self):
        try:
            return self.frame_queue.popleft()  # 큐에서 가장 최신 프레임을 가져옴
        except IndexError:
            return None  # 큐가 비어있으면 None을 반환

    def stop(self):
        self.running = False

# =================================================================
# 2. 6D Pose 추론을 위한 PoseCNN 모델 정의 (기존과 동일)
# =================================================================
class PoseCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = 512
        self.pose_head = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(256, 6))
        self.class_head = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(128, num_classes))
    def forward(self, x):
        f = self.backbone(x).flatten(1)
        pose_out = self.pose_head(f)
        cls_out = self.class_head(f)
        return pose_out, cls_out

# =================================================================
# 3. 메인 로직을 수행하는 새로운 ROS2 노드 클래스
# =================================================================
class VisionCoordinatorNode(Node):
    def __init__(self, video_receiver):
        super().__init__("pose_predictor_test")
        self.get_logger().info("Initializing Vision Coordinator Node...")

        # --- 상태 및 멤버 변수 초기화 ---
        self.state = 'IDLE'
        self.receiver = video_receiver
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.real_cur_cord = None
        self.target_object_name = "12"
        self.yolo_timer = None
        self.cnn_servoing_timer = None
        
        # --- PID 제어기 파라미터 및 변수 ---
        self.KP = 0.4
        self.KI = 0.007
        self.KD = 0.05  # D 제어기 게인 추가
        self.CONVERGENCE_THRESHOLD = 3
        self.integral_error = np.zeros(6, dtype=np.float32) # I 제어를 위한 이전 에러
        self.previous_error = np.zeros(6, dtype=np.float32) # D 제어를 위한 이전 에러
        self.last_servoing_time = None
        self.integral_clamp = 2.0

        # --- 모델 및 리소스 경로 설정 ---
        resource_path = "/home/addinedu/roscamp-repo-1/shopee_ros2/src/pickee_vision/resource"
        yolo_model_path = os.path.join(resource_path, '20251104_v11_ver1_ioudefault.pt')
        cnn_model_path = os.path.join(resource_path, "20251112_total.pt")
        target_image_path = os.path.join(resource_path, "test/target_fish_4.jpg")

        # --- 역정규화 파라미터 ---
        self.pose_mean = np.array([-75.24822998046875, 140.6298370361328, 220.1119842529297, -179.412109375, 0.4675877094268799, 44.999176025390625], dtype=np.float32)
        self.pose_std = np.array([31.43274688720703, 30.908634185791016, 1.6736514568328857, 0.38159045577049255, 1.5441224575042725, 25.34377098083496], dtype=np.float32)

        # --- 모델 로드 (YOLO & CNN) ---
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.cnn_model = PoseCNN(num_classes=2).to(self.device)
            self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
            self.cnn_model.eval()
            self.get_logger().info("YOLO and CNN models loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load models: {e}")
            raise e

        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])

        # --- 목표 이미지 Pose 추론 및 역정규화 (최초 1회) ---
        try:
            target_image = cv2.imread(target_image_path)
            norm_tar_cord = self.predict_pose(target_image)
            self.tar_cord = self.de_standardize_pose(norm_tar_cord)
            self.get_logger().info(f"Target pose predicted and denormalized: {self.tar_cord}")
        except Exception as e:
            self.get_logger().error(f"Failed to predict target pose: {e}")
            raise e

        # --- ROS2 통신 인터페이스 설정 ---
        # 1. main 노드로부터 피킹 시작 명령을 받는 서비스 서버
        self.start_pick_srv = self.create_service(ArmPickProduct, '/pickee/arm/pick_product', self.start_pick_sequence_callback)
        
        # 2. arm 노드에 매대 확인 자세를 요청하는 서비스 클라이언트
        self.move_start_client = self.create_client(Trigger, '/pickee/arm/move_start')
        while not self.move_start_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arm move_start service not available, waiting...')

        # 3. arm 노드가 top_view 자세에 도달했음을 알리는 토픽을 구독
        self.arm_ready_sub = self.create_subscription(Bool, '/pickee/arm/is_moving', self.arm_ready_callback, 10)

        # 4. arm 노드로부터 실제 좌표를 받는 서브스크라이버
        self.pose_subscriber = self.create_subscription(Pose6D, '/pickee/arm/real_pose', self.real_pose_callback, 10)

        # 5. CNN 서보잉 제어 명령을 보내는 퍼블리셔
        self.move_publisher = self.create_publisher(Pose6D, '/pickee/arm/move_servo', 10)

        # 6. 최종 그리핑 명령을 보내는 서비스 클라이언트
        self.grep_product_client = self.create_client(Trigger, '/pickee/arm/grep_product')
        while not self.grep_product_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Grep product service not available, waiting...')

        self.get_logger().info("Vision Coordinator Node started. Current state: IDLE")

    def set_state(self, new_state):
        self.get_logger().info(f"State transition: {self.state} -> {new_state}")
        self.state = new_state

    # --- Service and Topic Callbacks ---

    def start_pick_sequence_callback(self, request, response):
        if self.state != 'IDLE':
            response.success = False
            response.message = f"Node is not in IDLE state, current state: {self.state}"
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info("Start pick sequence request received from main.")
        self.set_state('WAITING_FOR_SHELF_VIEW')

        # [수정] /pickee/arm/move_start Trigger 서비스 호출
        arm_request = Trigger.Request()
        future = self.move_start_client.call_async(arm_request)
        future.add_done_callback(self.move_start_response_callback)

        response.success = True
        response.message = "Request accepted. Asking arm to move to shelf_view via move_start service."
        return response

    def move_start_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Arm successfully started move to shelf_view. Starting YOLO detection.")
                self.set_state('YOLO_DETECTION')
                if self.yolo_timer is None or self.yolo_timer.is_canceled():
                    self.yolo_timer = self.create_timer(0.03, self.yolo_detection_loop)
            else:
                self.get_logger().error(f"Arm failed to start move_start: {response.message}")
                self.set_state('IDLE')
        except Exception as e:
            self.get_logger().error(f"Service call to /pickee/arm/move_start failed: {e}")
            self.set_state('IDLE')

    def arm_ready_callback(self, msg):
        """[수정] /pickee/arm/is_moving 토픽 콜백"""
        if msg.data is True and self.state == 'WAITING_FOR_TOP_VIEW':
            self.get_logger().info("Arm is ready for CNN servoing. Starting...")
            # PID 제어기 상태 초기화
            self.integral_error = np.zeros(6, dtype=np.float32)
            self.previous_error = np.zeros(6, dtype=np.float32)
            self.last_servoing_time = None
            self.set_state('CNN_SERVOING')
            if self.cnn_servoing_timer is None or self.cnn_servoing_timer.is_canceled():
                self.cnn_servoing_timer = self.create_timer(0.03, self.cnn_servoing_loop)
    
    def real_pose_callback(self, msg):
        self.real_cur_cord = np.array([msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz])

    def grep_product_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Successfully called grep_product service: {response.message}")
            else:
                self.get_logger().error(f"Failed to call grep_product service: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Grep service call failed with exception: {e}")

    # --- Main Logic Loops (called by timers) ---

    def yolo_detection_loop(self):
        if self.state != 'YOLO_DETECTION': return
        frame = self.receiver.get_frame()
        if frame is None: print("캠 안들어와!!!!!!!!!!!!!!!!!!!!!!!"); return
        results = self.yolo_model(frame, conf=0.8, device=self.device, verbose=False)
        target_found = False
        if results:
            for result in results:
                if not hasattr(result, 'boxes'): continue
                for box in result.boxes:
                    cls_id = int(box.cls)
                    if result.names[cls_id] == self.target_object_name:
                        target_found = True
                        break
                if target_found: break
        if target_found:
            self.get_logger().info(f"Target '{self.target_object_name}' detected with YOLO.")
            if self.yolo_timer: self.yolo_timer.cancel()
            

            '''
            통합해야 할 부분 - Pickee Main에 YOLO 객체 인식 결과 반환
            '''

            
            self.get_logger().info("YOLO detection finished. Waiting for arm to move to top_view_pose.")
            self.set_state('WAITING_FOR_TOP_VIEW')
        
        cv2.imshow("YOLO Detection", results[0].plot())
        cv2.waitKey(1)

    def cnn_servoing_loop(self):
        if self.state != 'CNN_SERVOING': return
        frame = self.receiver.get_frame()
        if frame is None or self.real_cur_cord is None: return

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

        if error_magnitude < self.CONVERGENCE_THRESHOLD:
            self.get_logger().info(f"Target reached. Calling grep_product service.")
            if self.cnn_servoing_timer: self.cnn_servoing_timer.cancel()
            
            # [수정] Trigger 서비스를 호출하여 피킹 동작 요청
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

            # PI 제어 (X, Y, Z, RX, RY)
            delta = proportional_term + integral_term
            commanded_pose = self.real_cur_cord + delta
            
            # PD 제어 (RZ - Yaw)
            commanded_pose[5] = self.real_cur_cord[5] + proportional_term[5] + derivative_term[5]
            
            print(real_tar_cord) #####
            print(self.real_cur_cord)
            move_cmd = Pose6D()
            
            move_cmd.x, move_cmd.y, move_cmd.rz = float(commanded_pose[0]), float(commanded_pose[1]), float(commanded_pose[5])
            move_cmd.rx, move_cmd.ry, move_cmd.z = float(self.tar_cord[3]), float(self.tar_cord[4]), float(self.tar_cord[2])
            
            # move_cmd.x, move_cmd.y = float(commanded_pose[0]), float(commanded_pose[1])
            # move_cmd.rx, move_cmd.ry, move_cmd.z, move_cmd.rz = float(self.tar_cord[3]), float(self.tar_cord[4]), float(self.tar_cord[2]), float(self.tar_cord[5])
            
            self.move_publisher.publish(move_cmd)
            time.sleep(2.3)
        
        cv2.imshow("CNN Servoing", frame)
        cv2.waitKey(1)

    # --- Helper and Cleanup Methods ---

    def predict_pose(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pose_pred, _ = self.cnn_model(img_tensor)
            return pose_pred.cpu().numpy().flatten()

    def de_standardize_pose(self, norm_pose):
        return (norm_pose * self.pose_std) + self.pose_mean

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

# =================================================================
# 4. 메인 실행 함수
# =================================================================
def main(args=None):
    rclpy.init(args=args)
    video_receiver = VideoReceiver()
    video_thread = threading.Thread(target=video_receiver.run, daemon=True)
    video_thread.start()
    
    node = VisionCoordinatorNode(video_receiver)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
        video_receiver.stop()
        video_thread.join()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
