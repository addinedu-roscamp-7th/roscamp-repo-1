import rclpy
from rclpy.node import Node

# Import service and message types from shopee_interfaces
from shopee_interfaces.srv import PickeeArmMoveToPose, PickeeArmPickProduct, PickeeArmPlaceProduct
from shopee_interfaces.msg import ArmPoseStatus, PickeeArmTaskStatus

class PickeeArmController(Node):
    """
    Handles service requests to control the Pickee robot arm and publishes status updates.
    """
    def __init__(self):
        super().__init__('pickee_arm_controller')

        # Create Service Servers
        self.move_to_pose_srv = self.create_service(
            PickeeArmMoveToPose, '/pickee/arm/move_to_pose', self.move_to_pose_callback)
        
        self.pick_product_srv = self.create_service(
            PickeeArmPickProduct, '/pickee/arm/pick_product', self.pick_product_callback)

        self.place_product_srv = self.create_service(
            PickeeArmPlaceProduct, '/pickee/arm/place_product', self.place_product_callback)

        # Create Publishers
        self.pose_status_pub = self.create_publisher(
            ArmPoseStatus, '/pickee/arm/pose_status', 10)
        
        self.pick_status_pub = self.create_publisher(
            PickeeArmTaskStatus, '/pickee/arm/pick_status', 10)

        self.place_status_pub = self.create_publisher(
            PickeeArmTaskStatus, '/pickee/arm/place_status', 10)

        self.get_logger().info('Pickee Arm Controller Node has been started.')

    def move_to_pose_callback(self, request, response):
        """Callback for the /pickee/arm/move_to_pose service."""
        self.get_logger().info(
            f'Move to pose request received: order_id={request.order_id}, pose_type={request.pose_type}')

        # TODO: Implement actual arm movement logic
        
        # Publish a "completed" status immediately for now
        status_msg = ArmPoseStatus()
        status_msg.robot_id = request.robot_id
        status_msg.order_id = request.order_id
        status_msg.pose_type = request.pose_type
        status_msg.status = "completed"
        status_msg.progress = 1.0
        status_msg.message = f"Reached {request.pose_type} pose"
        self.pose_status_pub.publish(status_msg)
        
        response.success = True
        response.message = "Move to pose command accepted"
        return response

    def pick_product_callback(self, request, response):
        """Callback for the /pickee/arm/pick_product service."""
        self.get_logger().info(
            f'Pick product request received: order_id={request.order_id}, product_id={request.target_product.product_id}')

        # TODO: Implement actual picking logic
        
        # Publish a "completed" status immediately for now
        status_msg = PickeeArmTaskStatus()
        status_msg.robot_id = request.robot_id
        status_msg.order_id = request.order_id
        status_msg.product_id = request.target_product.product_id
        status_msg.status = "completed"
        status_msg.current_phase = "done"
        status_msg.progress = 1.0
        status_msg.message = "Product picked successfully"
        self.pick_status_pub.publish(status_msg)

        response.accepted = True
        response.message = "Pick command accepted"
        return response

    def place_product_callback(self, request, response):
        """Callback for the /pickee/arm/place_product service."""
        self.get_logger().info(
            f'Place product request received: order_id={request.order_id}, product_id={request.product_id}')

        # TODO: Implement actual placing logic

        # Publish a "completed" status immediately for now
        status_msg = PickeeArmTaskStatus()
        status_msg.robot_id = request.robot_id
        status_msg.order_id = request.order_id
        status_msg.product_id = request.product_id
        status_msg.status = "completed"
        status_msg.current_phase = "done"
        status_msg.progress = 1.0
        status_msg.message = "Product placed in cart successfully"
        self.place_status_pub.publish(status_msg)

        response.accepted = True
        response.message = "Place command accepted"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PickeeArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()