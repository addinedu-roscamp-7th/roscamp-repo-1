from example_interfaces.srv import AddTwoInts
from shopee_interfaces.srv import PickeeMobileMoveToLocation

import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(PickeeMobileMoveToLocation, '/pickee/mobile/move_to_location', self.add_two_ints_callback)# 서비스 자료형, 서비스 이름, 실행할 콜백 함수

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))

        return response


def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()