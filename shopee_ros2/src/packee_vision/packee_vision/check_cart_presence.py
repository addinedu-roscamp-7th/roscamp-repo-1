import rclpy 
from rclpy.node import Node
from shopee_interfaces.srv import PackeeVisionCheckCartPresence


class CheckCart(Node):

    def __init__(self):
        super().__init__("check_cart_presence")
        self.server = self.create_service(
            PackeeVisionCheckCartPresence,
            "check_cart_presence", # 타입명
            self.callback_service
        )
    
    def callback_service(self, request, response):
        self.get_logger().info(f"Received request for robot_id: {request.robot_id}") 

        return response


def main(args=None):
    rclpy.init(args=args)
    check_cart = CheckCart()
    rclpy.spin(check_cart)
    rclpy.shutdown()


if __name__=="__main__":
    main()