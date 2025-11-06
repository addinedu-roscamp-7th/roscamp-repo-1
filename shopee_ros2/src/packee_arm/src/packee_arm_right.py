import rclpy
from rclpy.node import Node
import time
from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, PackeeVisionDetectProductsInCart
import numpy as np
from pymycobot.mycobot280 import MyCobot280


class PackeeArmController(Node):
    def __init__(self):
        super().__init__("Packee_Arm_Controller")

        self.speed = 30
        
        ## 서비스 서버
        self.move_to_pose_server = self.create_service(ArmMoveToPose, "/packee1/arm/move_to_pose", self.move_to_arm)
        self.pickup_server = self.create_service(ArmPickProduct, "/packee1/arm/pick_product", self.pickup_product)

        ## 서비스 client
        self.vision_client = self.create_client(PackeeVisionDetectProductsInCart, "/packee1/vision/detect_products_in_cart")

        self.gain = 0.3
        self.epsilon = 10.0
        self.products = {1: "와사비", 12: "생선", 14: "이클립스"}

        self.mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
        self.mc.thread_lock = True
        self.get_logger().info("로봇이 연결되었습니다.")

    def move_to_arm(self, request: ArmMoveToPose.Request,response: ArmMoveToPose.Response):
        robot_id = request.robot_id
        order_id = request.order_id
        pose_type = request.pose_type
        self.get_logger().info(f"request Data: robot_id={robot_id}, order_id={order_id}, pose_type={pose_type}")

        try:
            self.get_logger().info("packee1 Arm 자세변경을 수락했습니다.")
            if pose_type == "cart_view":
                self.mc.send_coords([42.2, -39.0, 289.8, -153.04, 21.75, -85.67], self.speed)
            else:
                self.mc.send_angles([0, 0, 0, 0, 0, 0], self.speed)
            time.sleep(1)

            self.get_logger().info("packee1 Arm 자세변경을 완료했습니다.")

            response.success = True
            response.message = "packee1 Arm 자세변경 완료"

        except Exception as e:
            self.get_logger().info("packee1 Arm 자세변경을 실패했습니다.")

            response.success = False
            response.message = "packee1 Arm 자세변경 실패"

        return response
    
    def call_vision_service(self, robot_id, order_id, product_id):
        while not self.vision_client.wait_for_service(10):
            self.get_logger().warn("waiting for service ...")

        request = PackeeVisionDetectProductsInCart()
        request.robot_id = robot_id
        request.order_id = order_id
        request.expected_product_id = product_id

        future = self.vision_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()


    def pickup_product(self, request:ArmPickProduct.Request, response:ArmPickProduct.Response):
        robot_id = request.robot_id
        order_id = request.order_id
        product_id = request.product_id
        arm_side = request.arm_side
        pose = request.pose
        self.get_logger().info(f"pickup Request Data: robot_id={robot_id}, order_id={order_id}, product_id={product_id}, arm_side={arm_side}, pose={pose}")

        try:
            self.mc.send_coords([42.2, -39.0, 289.8, -153.04, 21.75, -85.67], self.speed)
            self.get_logger().info(f"packee1 Arm 장바구니 확인자세 변경합니다.")
            time.sleep(1)

            complete_flag = False
            while not complete_flag:
                results = self.call_vision_service(robot_id, order_id, product_id)
                current_pose = results["current_pose"]
                target_pose = results["target_pose"]

                delta_pose = self.gain * (target_pose - current_pose)
                pose = current_pose + delta_pose

                self.get_logger().info(f"packee1 Arm을 x={pose[0]}, y={pose[1]}, z={pose[2]}, rx={pose[3]}, ry={pose[4]}, rz={pose[5]} 만큼 이동합니다.")
                self.mc.send_coords(pose.to_list(), self.speed)
                
                error_norm = np.linalg.norm(target_pose - pose)

                if error_norm < self.epsilon:
                    self.get_logger().info("목표 포즈에 도달했습니다. 상품을 픽업합니다.")
                    lift_pose = pose.copy()
                    lift_pose[2] -= 40.0
                    self.mc.send_coords(lift_pose.to_list(), self.speed)
                    time.sleep(1)

                    self.mc.set_gripper_value(0, self.speed)
                    self.get_logger().info("그리퍼를 닫습니다.")
                    time.sleep(1)

                    self.mc.send_angles([0, 0, 0, 0, 0, 0], self.speed)
                    self.get_logger().info("상품을 들어올립니다.")
                    complete_flag=True

            response.success = True
            response.message = f"{self.products[product_id]} 상품 픽업 완료"

        except Exception as e:
            response.success = False
            response.message = f"{self.products[product_id]} 상품 픽업 실패"
            self.get_logger().error("packee1 상품 픽업을 실패했습니다.")

        return response
    
def main():
    rclpy.init()

    node = PackeeArmController()

    try:
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
