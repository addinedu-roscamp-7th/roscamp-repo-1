import rclpy 
from rclpy.node import Node
from shopee_interfaces.srv import PackeeVisionCheckCartPresence


class CheckCart(Node):

    def __init__(self):
        super().__init__("check_cart_presence")
        self.server = self.create_service(
            PackeeVisionCheckCartPresence,
            "packee/vision/check_cart_presence", # 타입명
            self.callback_service
        )
    
    def callback_service(self, request, response):
        self.get_logger().info(f"Received request for robot_id: {request.robot_id}") 

        response.cart_present = True
        response.confidence = 0.93
        response.message = 'good'

        self.get_logger().info(
        f"Response -> cart_present: {response.cart_present}, "
        f"confidence: {response.confidence}, message: {response.message}"
    )


        return response


def main(args=None):
    rclpy.init(args=args)
    check_cart = CheckCart()
    rclpy.spin(check_cart)
    rclpy.shutdown()


if __name__=="__main__":
    main()