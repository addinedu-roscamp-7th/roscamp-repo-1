import rclpy
from rclpy.node import Node

# Import service and message types from shopee_interfaces
from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, ArmPlaceProduct
from shopee_interfaces.msg import ArmPoseStatus, ArmTaskStatus

# Import the new modules
from .kinematics_model import KinematicsModel
from .visual_servoing_controller import VisualServoingController

class PickeeArmController(Node):
    """
    Handles service requests to control the Pickee robot arm and publishes status updates.
    """
    def __init__(self):
        super().__init__('pickee_arm_controller')

        # Instantiate the new components
        self.kinematics = KinematicsModel(self.get_logger())
        self.servoing_controller = VisualServoingController(self.get_logger(), self.kinematics)

        # Create Service Servers
        self.move_to_pose_srv = self.create_service(
            ArmMoveToPose, '/pickee/arm/move_to_pose', self.move_to_pose_callback)
        
        self.pick_product_srv = self.create_service(
            ArmPickProduct, '/pickee/arm/pick_product', self.pick_product_callback)

        self.place_product_srv = self.create_service(
            ArmPlaceProduct, '/pickee/arm/place_product', self.place_product_callback)

        # Create Publishers
        self.pose_status_pub = self.create_publisher(
            ArmPoseStatus, '/pickee/arm/pose_status', 10)
        
        self.pick_status_pub = self.create_publisher(
            ArmTaskStatus, '/pickee/arm/pick_status', 10)

        self.place_status_pub = self.create_publisher(
            ArmTaskStatus, '/pickee/arm/place_status', 10)

        self.get_logger().info('Pickee Arm Controller Node has been started.')

    def move_to_pose_callback(self, request, response):
        """Callback for the /pickee/arm/move_to_pose service."""
        self.get_logger().info(
            f'Move to pose request received: order_id={request.order_id}, pose_type={request.pose_type}')

        # TODO: Implement actual arm movement logic using IK from kinematics_model
        # 1. Get target pose for the given pose_type (e.g., from a config file)
        # 2. Call self.kinematics.inverse_kinematics(target_pose)
        # 3. Send the resulting joint states to the robot driver

        self.get_logger().info(f"Simulating move to {request.pose_type}...")
        
        # Publish a "completed" status immediately for now
        status_msg = ArmPoseStatus()
        status_msg.robot_id = request.robot_id
        status_msg.order_id = request.order_id
        status_msg.pose_type = request.pose_type
        status_msg.status = "completed"
        status_msg.progress = 1.0
        status_msg.message = f"Reached {request.pose_type} pose (simulation)"
        self.pose_status_pub.publish(status_msg)
        
        response.success = True
        response.message = "Move to pose command accepted"
        return response

    def pick_product_callback(self, request, response):
        """Callback for the /pickee/arm/pick_product service."""
        self.get_logger().info(
            f'Pick product request received: order_id={request.order_id}, product_id={request.target_product.product_id}')

        # The core logic will now use the servoing controller.
        # This would typically run in a separate thread or async task.
        self.get_logger().info("Visual servoing process would start here.")

        # TODO: Implement the full picking process loop:
        # 1. Publish "in_progress" with "approaching" phase.
        # 2. Get current arm pose and target pose.
        # 3. Loop:
        #    - joint_velocities = self.servoing_controller.calculate_velocity_command(current, target)
        #    - Send velocities to robot driver.
        #    - Update current pose.
        #    - If close enough, break loop.
        # 4. Grasp the object.
        # 5. Publish "completed" status.
        
        # For now, we just log the intent and publish completion.
        status_msg = ArmTaskStatus()
        status_msg.robot_id = request.robot_id
        status_msg.order_id = request.order_id
        status_msg.product_id = request.target_product.product_id
        status_msg.status = "completed"
        status_msg.current_phase = "done"
        status_msg.progress = 1.0
        status_msg.message = "Product picked successfully (simulation)"
        self.pick_status_pub.publish(status_msg)

        response.accepted = True
        response.message = "Pick command accepted"
        return response

    def place_product_callback(self, request, response):
        """Callback for the /pickee/arm/place_product service."""
        self.get_logger().info(
            f'Place product request received: order_id={request.order_id}, product_id={request.product_id}')

        # TODO: Implement placing logic, likely using IK from kinematics_model.
        # 1. Get target pose for the cart.
        # 2. Call self.kinematics.inverse_kinematics(cart_pose).
        # 3. Send joint states to robot driver.
        # 4. Open gripper.

        self.get_logger().info("Placing process would start here.")

        # Publish a "completed" status immediately for now
        status_msg = ArmTaskStatus()
        status_msg.robot_id = request.robot_id
        status_msg.order_id = request.order_id
        status_msg.product_id = request.product_id
        status_msg.status = "completed"
        status_msg.current_phase = "done"
        status_msg.progress = 1.0
        status_msg.message = "Product placed in cart successfully (simulation)"
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
