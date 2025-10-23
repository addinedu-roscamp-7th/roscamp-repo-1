import rclpy
from rclpy.node import Node
import random

from shopee_interfaces.msg import PickeeVisionObstacles, Obstacle, Point2D, Vector2D, BBox

class ObstacleDetectorNode(Node):
    """
    주기적으로 가상의 장애물 데이터를 생성하고 발행하는 노드입니다.
    """
    def __init__(self):
        super().__init__('obstacle_detector_node')
        self.publisher_ = self.create_publisher(
            PickeeVisionObstacles,
            '/pickee/vision/obstacle_detected',
            10)
        
        # 2초마다 publish_obstacles 메소드 실행
        self.timer = self.create_timer(2.0, self.publish_obstacles)
        self.get_logger().info('Obstacle Detector Node has been started.')

    def publish_obstacles(self):
        """가상의 장애물 데이터를 생성하고 토픽으로 발행합니다."""
        msg = PickeeVisionObstacles()
        msg.robot_id = 1
        msg.order_id = 999 # 테스트용 order_id

        # 가상의 장애물 1개 생성
        obstacle = Obstacle()
        obstacle.obstacle_type = "cart" # or "person", "box"
        obstacle.position = Point2D(x=random.uniform(3.0, 5.0), y=random.uniform(-1.0, 1.0))
        obstacle.distance = float(obstacle.position.x)
        obstacle.velocity = 0.0
        obstacle.direction = Vector2D(vx=0.0, vy=0.0)
        obstacle.bbox = BBox(x1=200, y1=150, x2=350, y2=400)
        obstacle.confidence = random.uniform(0.9, 0.99)

        msg.obstacles.append(obstacle)
        msg.message = f"{len(msg.obstacles)} static obstacle(s) detected. 안녕하세요."

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing obstacle data: type={obstacle.obstacle_type}, dist={obstacle.distance:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
