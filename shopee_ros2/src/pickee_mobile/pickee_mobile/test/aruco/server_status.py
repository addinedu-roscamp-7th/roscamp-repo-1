#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from shopee_interfaces.srv import PickeeMobileStatus


class PickeeMobileStatusServer(Node):
    def __init__(self):
        super().__init__('pickee_mobile_status_server')

        self.srv = self.create_service(
            PickeeMobileStatus,
            'pickee/mobile/pickee_mobile_status',
            self.service_callback
        )

        self.get_logger().info("✅ PickeeMobileStatus service server started!")

    def service_callback(self, request, response):
        robot_id = request.robot_id
        status = request.status

        self.get_logger().info(
            f"📩 Received docking status | robot_id={robot_id}, status='{status}'"
        )

        # 여기에서 상태 정보를 DB 저장, 로그 저장, FSM 으로 전달 등등 하면 됨
        # 현재는 성공 응답만 돌려줌
        response.success = True
        response.message = f"Received status '{status}' from robot {robot_id}"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PickeeMobileStatusServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
