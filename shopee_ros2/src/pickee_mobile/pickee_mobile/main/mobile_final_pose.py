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

from pickee_mobile.main.mobile_go_strait import run
from pickee_mobile.main.mobile_rotate import rotate