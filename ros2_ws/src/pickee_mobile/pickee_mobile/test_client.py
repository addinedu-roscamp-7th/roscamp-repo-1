import sys
from shopee_interfaces.srv import PickeeMobileMoveToLocation

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(PickeeMobileMoveToLocation, '/pickee/mobile/move_to_location') # 서비스 명, 해당 서비스로 메시지 전송
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info('service available')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        return self.cli.call_async(self.req)


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()
    
    try:
        future = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
        rclpy.spin_until_future_complete(minimal_client, future)
        response = future.result()
        minimal_client.get_logger().info(
            'Result of add_two_ints: for %d + %d = %d' %
            (int(sys.argv[1]), int(sys.argv[2]), response.sum))
    
    except:
        minimal_client.get_logger().info('!!!please type 2 numbers!!!')

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()