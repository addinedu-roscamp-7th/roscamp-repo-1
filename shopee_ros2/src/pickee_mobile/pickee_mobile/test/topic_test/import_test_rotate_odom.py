#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import math
import time
from geometry_msgs.msg import Twist

# rotate 함수 불러오기
# from pickee_mobile.module.module_rotate import rotate
from pickee_mobile.module.module_rotate_odom import Rotate


class RotateTest(Node):
    def __init__(self):
        super().__init__("rotate_test_node")

        # 노드 초기화되면 바로 테스트 회전 수행 (타이머 1회)
        self.create_timer(1.0, self.run_once)
        self.executed = False

        self.get_logger().info("✅ RotateTest node started. Will rotate shortly...")
        self.pub = self.create_publisher(Twist, '/cmd_vel_modified', 10)
        self.node = Rotate()


    def run_once(self):
        if self.executed:
            return
        self.executed = True
        

        self.get_logger().info("🔁 Calling rotate(self, 90°)")
        self.node.rotate(math.radians(90))

        time.sleep(2.0)
        self.node.rotate(math.radians(-90))

        self.get_logger().info("✅ Rotate test complete!")


def main(args=None):
    rclpy.init(args=args)
    node = RotateTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
