# /home/addinedu/roscamp-repo-1/shopee_ros2/src/pickee_arm/scripts/inference.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped # 추론 결과를 발행할 토픽 (예시)
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import os
import numpy as np

# 1. 모델 정의 (Model Definition) - train.py와 동일해야 합니다.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50) # 입력 특성 10개, 은닉층 50개
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)  # 출력 클래스 2개 (예: 픽업 성공/실패)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.get_logger().info('Inference Node has been started.')

        self.bridge = CvBridge()

        # 모델 로드 경로 설정
        # 라즈베리파이에서 모델 파일이 위치할 경로를 지정합니다.
        # PC에서 학습된 'simple_nn_model.pth' 파일을 이 경로로 수동으로 옮겨야 합니다.
        self.model_path = '/home/addinedu/pickee_arm_models/simple_nn_model.pth' # 예시 경로
        # 실제 사용 시에는 라즈베리파이의 적절한 경로로 변경해주세요.
        # 예: os.path.join(os.path.expanduser('~'), 'pickee_arm_models', 'simple_nn_model.pth')

        # 모델 로드
        self.model = SimpleNN()
        if os.path.exists(self.model_path):
            # CPU로 로드 (라즈베리파이는 보통 GPU가 없으므로)
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval() # 추론 모드로 설정
            self.get_logger().info(f"Model loaded successfully from {self.model_path}")
        else:
            self.get_logger().error(f"Model file not found at {self.model_path}. Please transfer the trained model.")
            self.model = None # 모델 로드 실패 시 None으로 설정

        # 이미지 구독자 설정
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw', # 실제 카메라 이미지 토픽으로 변경하세요
            self.image_callback,
            10
        )
        self.image_subscription # prevent unused variable warning

        # 추론 결과 발행자 설정 (예: 로봇 팔의 목표 포즈)
        self.pose_publisher = self.create_publisher(PoseStamped, '/pickee_arm/target_pose', 10)

        self.get_logger().info("Inference node ready to receive images.")

    def image_callback(self, msg):
        if self.model is None:
            self.get_logger().warn("Model not loaded. Skipping inference.")
            return

        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # self.get_logger().info(f"Received image of shape: {cv_image.shape}")

            # 여기에 이미지 전처리 로직을 추가합니다.
            # 예: 크기 조정, 정규화, 텐서 변환 등
            # 현재 SimpleNN은 10개의 특성을 입력으로 받으므로,
            # 이미지에서 10개의 특성을 추출하는 과정이 필요합니다.
            # 이 예시에서는 임의의 10개 특성을 생성하여 사용합니다.
            # 실제 구현에서는 이미지 처리(CNN 등)를 통해 특성을 추출해야 합니다.
            processed_input = self.preprocess_image_for_model(cv_image)

            if processed_input is None:
                self.get_logger().warn("Image preprocessing failed. Skipping inference.")
                return

            # PyTorch 텐서로 변환
            input_tensor = torch.from_numpy(processed_input).float().unsqueeze(0) # 배치 차원 추가

            # 추론 수행
            with torch.no_grad():
                output = self.model(input_tensor)

            # 추론 결과 처리 (예: 가장 높은 확률의 클래스 선택)
            _, predicted_class = torch.max(output, 1);
            self.get_logger().info(f"Inference result: Predicted class = {predicted_class.item()}")

            # 추론 결과를 ROS 토픽으로 발행 (예시: PoseStamped 메시지)
            # 실제 모델의 출력에 따라 PoseStamped 메시지를 구성하는 로직이 필요합니다.
            # 여기서는 예시로 간단한 메시지를 발행합니다.
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'base_link' # 적절한 프레임 ID로 변경
            pose_msg.pose.position.x = float(predicted_class.item()) * 0.1 # 예시 값
            pose_msg.pose.position.y = 0.0
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0 # 예시 값 (회전 없음)
            self.pose_publisher.publish(pose_msg)
            self.get_logger().info(f"Published target pose: x={pose_msg.pose.position.x}")

        except Exception as e:
            self.get_logger().error(f"Error during image processing or inference: {e}")

    def preprocess_image_for_model(self, cv_image):
        # 이 함수는 실제 모델의 입력 요구사항에 맞게 이미지를 전처리해야 합니다.
        # 현재 SimpleNN은 10개의 숫자 특성을 입력으로 받습니다.
        # 따라서 이미지에서 10개의 의미 있는 특성을 추출하는 로직이 필요합니다.
        # 예시: 이미지의 평균 픽셀 값, 특정 영역의 특징 등
        # 여기서는 임의의 10개 특성을 반환하는 것으로 대체합니다.
        # 실제 딥러닝 모델(예: CNN)을 사용한다면, 이미지 크기 조정, 정규화 후 텐서로 변환하는 로직이 들어갑니다.
        self.get_logger().warn("Using dummy image preprocessing. Implement actual feature extraction from image.")
        return np.random.rand(10).astype(np.float32) # 임의의 10개 특성 반환

def main(args=None):
    rclpy.init(args=args)
    inference_node = InferenceNode()
    rclpy.spin(inference_node)
    inference_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()