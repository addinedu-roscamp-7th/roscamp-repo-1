import rclpy
import cv2
import math
from rclpy.node import Node
from pickee_mobile.sub.sub_aruco_detect import ArucoPoseEstimator
from geometry_msgs.msg import Pose2D
from shopee_interfaces.msg import ArucoPose

class ArucoReaderNode(Node):
    def __init__(self):
        super().__init__('aruco_reader')
        self.get_logger().info("📷 ArUco Reader Node Started")

        # ArucoPoseEstimator 초기화
        self.estimator = ArucoPoseEstimator(
            camera_id=2,
            marker_length=50,  # mm 단위
            calibration_file="camera_calibration.pkl"
        )

        self.pose_publisher = self.create_publisher(ArucoPose, 
                                                    '/pickee/mobile/aruco_pose', 
                                                    10)

        # 0.2초마다 카메라에서 마커 읽기
        self.timer = self.create_timer(0.2, self.read_marker)

    def read_marker(self):
        ret, frame = self.estimator.cap.read()
        if not ret:
            self.get_logger().warning("❌ 프레임을 읽을 수 없습니다.")
            return

        frame_out, markers = self.estimator.process_frame(frame)

        for m in markers:
            self.get_logger().info(
                f"🟢 ID {m['id']} | x={m['x']:.1f}mm, y={m['y']:.1f}mm, z={m['z']:.1f}mm | "
                f"roll={m['roll']:.1f}°, pitch={m['pitch']:.1f}°, yaw={m['yaw']:.1f}°"
            )

        pose = ArucoPose()
        pose.x = markers[0]['x']
        pose.y = markers[0]['y']
        pose.z = markers[0]['z']
        pose.roll = markers[0]['roll']
        pose.pitch = markers[0]['pitch']
        pose.yaw = markers[0]['yaw']
        self.pose_publisher.publish(pose)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.estimator.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
