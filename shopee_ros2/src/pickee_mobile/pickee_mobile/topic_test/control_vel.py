import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Twist


class TwistModifier(Node):
    """Twist 메시지를 비율(scale)에 따라 수정하고 발행"""

    def __init__(self):
        super().__init__('twist_modifier')

        # 파라미터 선언
        self.declare_parameter('scale', 0.1)
        self.declare_parameter('robot_vel', '/cmd_vel')

        # 기존 /cmd_vel 구독
        self.create_subscription(Twist, '/cmd_vel', self.modify_cmd_vel_callback, 10)

        # 파라미터 기반 발행 토픽 설정
        topic_name = self.get_parameter('robot_vel').value
        
        vel_qos = QoSProfile(
                    depth=10,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.VOLATILE
                )
        self.pickee_vel_publisher = self.create_publisher(Twist, topic_name, vel_qos)

        self.get_logger().info(f'✅ Twist Modifier 시작됨 (scale={self.get_parameter("scale").value})')
        self.get_logger().info(f'출력 토픽: {topic_name}')

    def modify_cmd_vel_callback(self, msg: Twist):
        """속도를 비율(scale)에 따라 조정"""
        scale = self.get_parameter('scale').value
        new_msg = Twist()

        for axis in ['x', 'y', 'z']:
            setattr(new_msg.linear, axis, getattr(msg.linear, axis) * scale)
            setattr(new_msg.angular, axis, getattr(msg.angular, axis) * scale)

        


        self.pickee_vel_publisher.publish(new_msg)
        self.get_logger().info(
            f"Linear: ({msg.linear.x:.2f}, {msg.linear.y:.2f}) -> "
            f"({new_msg.linear.x:.2f}, {new_msg.linear.y:.2f}), "
            f"Angular Z: {msg.angular.z:.2f} -> {new_msg.angular.z:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TwistModifier()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('🛑 Twist Modifier 종료')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
