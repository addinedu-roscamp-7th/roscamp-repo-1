import rclpy 
from rclpy.node import Node
from shopee_interfaces.srv import PackeeVisionCheckCartPresence


class CheckCart(Node):

    def __init__(self):
        super().__init__("check_cart_presence")
        self.server = self.create_service(
            PackeeVisionCheckCartPresence,
            "packee/vision",
            self.callback_service
        )
    
    def callback_service(self, request, response)
        dfasdfas
    def main(args=None):
        rclpy.init(args=args)


if __name__=="main":
    main()
