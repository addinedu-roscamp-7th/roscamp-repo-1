import rclpy
from rclpy.node import Node 
from shopee_interfaces.srv import PackeeVisionVerifyPackingComplete

class PackingComplete(Node):
    def __init__(self):
        super().__init__("verify_packing_complete")
        self.server = self.create_service(
            PackeeVisionVerifyPackingComplete,
            "packee/vision/verify_packing_complete",
            self.callback_service
        )
    
    def callback_service(self, request, response):
        self.get_logger().info(f"Received request => robot_id: {request.robot_id} order_id: {request.order_id}") 

        response.cart_empty = True
        response.remaining_items = 0
        response.remaining_product_ids = []
        response.message = "Cart is empty, packing complete"

        return response



def main(args=None):
    rclpy.init(args=args)
    packing_complete = PackingComplete()
    rclpy.spin(packing_complete)
    rclpy.shutdown()



if __name__=="__main__":
    main()