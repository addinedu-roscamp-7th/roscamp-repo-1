import time
import rclpy
from pickee_mobile.module.module_go_strait import run
from pickee_mobile.module.module_rotate import rotate
from rclpy.node import Node

def main():
    rclpy.init()
    node = Node("import_go_strait_runner")

    print("🚀 go -0.5m")
    run(node, 0.5)

    print("🔄 rotate +30°")
    rotate(node, 30.0)

    time.sleep(1)

    print("🚀 go +0.5m")
    run(node, -0.5)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
