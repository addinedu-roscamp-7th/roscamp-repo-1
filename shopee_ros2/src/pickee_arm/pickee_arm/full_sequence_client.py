#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import asyncio
from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, ArmPlaceProduct
from shopee_interfaces.msg import DetectedProduct

class SequenceClient(Node):
    """Calls the pickee_arm services in a sequence to test the full motion."""

    def __init__(self):
        super().__init__('full_sequence_client')
        self.move_client = self.create_client(ArmMoveToPose, '/pickee/arm/move_to_pose')
        self.pick_client = self.create_client(ArmPickProduct, '/pickee/arm/pick_product')
        self.place_client = self.create_client(ArmPlaceProduct, '/pickee/arm/place_product')

    async def run_sequence(self):
        """서비스들을 순서대로 호출하여 전체 동작을 실행합니다."""
        
        for client in [self.move_client, self.pick_client, self.place_client]:
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f'{client.srv_name} service not available, waiting...')

        self.get_logger().info("--- 1. Calling Move to Standby Pose ---")
        move_request = ArmMoveToPose.Request(pose_type="standby")
        move_future = self.move_client.call_async(move_request)
        await move_future

        self.get_logger().info("--- 2. Calling Pick Service ---")
        pick_request = ArmPickProduct.Request(target_product=DetectedProduct(product_id=101))
        pick_future = self.pick_client.call_async(pick_request)
        await pick_future

        self.get_logger().info("--- 3. Calling Place Service ---")
        place_request = ArmPlaceProduct.Request(product_id=101)
        place_future = self.place_client.call_async(place_request)
        await place_future
            
        self.get_logger().info("--- Full Sequence Test Complete ---")

def main(args=None):
    rclpy.init(args=args)
    client_node = SequenceClient()
    asyncio.run(client_node.run_sequence())
    client_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()