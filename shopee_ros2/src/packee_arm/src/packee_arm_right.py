# 42.2, -39.0, 289.8, -153.04, 21.75, -85.67
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import time
from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, PackeeVisionDetectProductsInCart
import numpy as np
from pymycobot.mycobot280 import MyCobot280


class PackeeArmController(Node):
    def __init__(self):
        super().__init__("Packee_Arm_Controller")

        self.speed = 30
        self.gain = 0.3
        self.epsilon = 10.0
        self.products = {1: "와사비", 12: "생선", 14: "이클립스"}

        self.cb_group = ReentrantCallbackGroup()

        self.mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
        self.mc.thread_lock = True
        self.get_logger().info("로봇이 연결되었습니다.")

        self.move_to_pose_server = self.create_service(
            ArmMoveToPose,
            "/packee1/arm/move_to_pose",
            self.move_to_arm,
            callback_group=self.cb_group
        )

        self.pickup_server = self.create_service(
            ArmPickProduct,
            "/packee1/arm/pick_product",
            self.pickup_product,
            callback_group=self.cb_group
        )

        self.vision_client = self.create_client(
            PackeeVisionDetectProductsInCart,
            "/packee1/vision/detect_products_in_cart",
            callback_group=self.cb_group
        )

        self.get_logger().info("서비스 서버 생성 완료")

    def move_to_arm(self, request, response):
        pose_type = request.pose_type
        self.get_logger().info(f"Arm 자세변경 요청: pose_type={pose_type}")

        try:
            if pose_type == "cart_view":
                self.mc.send_coords([42.2, -39.0, 289.8, -153.04, 21.75, -85.67], self.speed)
            else:
                self.mc.send_angles([0, 0, 0, 0, 0, 0], self.speed)
            time.sleep(1)
            response.success = True
            response.message = "자세변경 완료"
        except Exception as e:
            self.get_logger().error(f"자세변경 실패: {e}")
            response.success = False
            response.message = f"자세변경 실패: {e}"
        return response

    def call_vision_service(self, robot_id, order_id, product_id):
        if not self.vision_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("비전 서비스 대기 타임아웃")
            return None

        req = PackeeVisionDetectProductsInCart.Request()
        req.robot_id = robot_id
        req.order_id = order_id
        req.expected_product_id = product_id

        future = self.vision_client.call_async(req)

        t0 = time.time()
        while rclpy.ok():
            if future.done():
                self.get_logger().info("비전 응답 수신 완료")
                return future.result()
            if time.time() - t0 > 10.0:
                self.get_logger().error("비전 응답 타임아웃")
                return None
            time.sleep(0.05)

    def pickup_product(self, request, response):
        robot_id = request.robot_id
        order_id = request.order_id
        product_id = request.product_id

        self.get_logger().info(f"상품 픽업 요청: product_id={product_id}")

        try:
            # 카트 확인 자세로 이동
            self.mc.send_coords([42.2, -39.0, 289.8, -153.04, 21.75, -85.67], self.speed)
            self.mc.set_gripper_value(100, self.speed)
            self.get_logger().info("장바구니 확인 자세로 이동")
            time.sleep(1)

            while True:
                results = self.call_vision_service(robot_id, order_id, product_id)
                if results is None or not results.success:
                    self.get_logger().warn("비전 응답 실패 또는 미탐지 → 재시도 중...")
                    continue

                cur = np.array([
                    results.current_pose.x, results.current_pose.y, results.current_pose.z,
                    results.current_pose.rx, results.current_pose.ry, results.current_pose.rz
                ])
                tgt = np.array([
                    results.target_pose.x, results.target_pose.y, results.target_pose.z,
                    results.target_pose.rx, results.target_pose.ry, results.target_pose.rz
                ])

                delta = (tgt - cur)
                pose = cur + delta

                self.get_logger().info(f"Arm 이동: {pose.tolist()}")
                self.mc.send_coords(pose.tolist(), self.speed)

                err = np.linalg.norm(tgt - pose)
                self.get_logger().info(f"pose error norm={err:.2f}")

                if err < self.epsilon:
                    self.get_logger().info("목표 포즈 도달 → 픽업 시작")

                    lift_pose = pose.copy()
                    lift_pose[2] -= 120.0
                    self.mc.send_coords(lift_pose.tolist(), self.speed)
                    time.sleep(1)

                    self.mc.set_gripper_value(0, self.speed)
                    self.get_logger().info("그리퍼 닫음")
                    time.sleep(1)

                    self.mc.send_angles([0, 0, 0, 0, 0, 0], self.speed)
                    self.get_logger().info("상품 들어올림 완료")

                    response.success = True
                    response.message = f"{self.products.get(product_id, '상품')} 픽업 완료"
                    return response

        except Exception as e:
            self.get_logger().error(f"픽업 실패: {e}")
            response.success = False
            response.message = f"픽업 실패: {e}"
            return response

def main():
    rclpy.init()
    node = PackeeArmController()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
