import rclpy 
from rclpy.node import Node
from shopee_interfaces.srv import PackeeVisionDetectProductsInCart
from shopee_interfaces.msg import PackeeDetectedProduct
from shopee_interfaces.msg import BBox
from shopee_interfaces.msg import Point3D



class DetectProducts(Node):
    def __init__(self):
        super().__init__("detect_products_in_cart")
        self.server = self.create_service(
            PackeeVisionDetectProductsInCart,
            "packee/vision/detect_products_in_cart",
            self.callback_service
        )
    
    def callback_service(self, request, response):
        self.get_logger().info(
            f"Received request -> "
            f"robot_id: {request.robot_id},"
            f"order_id: {request.order_id},"
            f"expend_product_ids: {list(request.expected_product_ids)}"
        )


        product_list = []

        p1 = PackeeDetectedProduct()
        p1.product_id = 3
        p1.confidence = 0.94
        p1.bbox = BBox(x1=120, y1=180, x2=250, y2=320)
        p1.position = Point3D(x=0.3, y=0.15, z=0.8)
        product_list.append(p1)

        p2 = PackeeDetectedProduct()
        p2.product_id = 4
        p2.confidence = 0.91
        p2.bbox = BBox(x1=280, y1=150, x2=380, y2=280)
        p2.position = Point3D(x=0.25, y=-0.1, z=0.75)
        product_list.append(p2)

        p3 = PackeeDetectedProduct()
        p3.product_id = 5
        p3.confidence = 0.89
        p3.bbox = BBox(x1=400, y1=200, x2=520, y2=340)
        p3.position = Point3D(x=0.2, y=0.2, z=0.7)
        product_list.append(p3)

        response.success = True
        response.products = product_list
        response.total_detected = len(product_list)
        response.message = "All products detected"

        self.get_logger().info(
            f"Response -> success={response.success}, total_detected={response.total_detected}, "
            f"message={response.message}"
        )

        return response

def main(args=None):
    rclpy.init(args=args)
    detect_product = DetectProducts()
    rclpy.spin(detect_product)
    rclpy.shutdown()

if __name__=="__main__":
    main()
