import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist

from shopee_interfaces.srv import PickeeMobileMoveToLocation
from shopee_interfaces.msg import PickeeMobileArrival, Pose2D, PickeeMobilePose

# from pickee_mobile.main.mobile_go_strait import run
# from pickee_mobile.main.mobile_rotate import rotate_inline

class GoToAruco(Node):
    def __init__(self):
        super().__init__('go_to_aruco_node')
        self.get_logger().info('🚀 GoToAruco 노드 시작 🚀')

        self._action_client = ActionClient(self, 
                                           NavigateToPose, 
                                           '/navigate_to_pose')

        self.create_service(PickeeMobileMoveToLocation, 
                            '/pickee/mobile/move_to_location', 
                            self.pickee_move_to_location_callback)
        
        self.arrival_publisher = self.create_publisher(PickeeMobileArrival, 
                                                       '/pickee/mobile/arrival',
                                                         10)
        
        self.pose_publisher = self.create_publisher(PickeeMobilePose,
                                                    '/pickee/mobile/pose',
                                                    10)
        
        self.vel_subscriber = self.create_subscription(Twist,
                                                       '/cmd_vel_modified',
                                                       self.vel_calculate_callback,
                                                       10)
        
        #초기 설정
        self.status = 'idle'
        self.working = 0 # 대기중

        self.pose_timer = self.create_timer(0.2, self.pose_publisher_timer_callback)
