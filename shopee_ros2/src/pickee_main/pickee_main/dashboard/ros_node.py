import rclpy
import re
from rclpy.node import Node
from rcl_interfaces.msg import Log
from shopee_interfaces.msg import PickeeRobotStatus, ArucoPose, PersonDetection
from PySide6.QtCore import QThread, Signal, QTimer


class RosNodeThread(QThread):
    log_received = Signal(str, str, str, bool)
    nodes_updated = Signal(list)
    topics_updated = Signal(list)
    services_updated = Signal(list)
    state_updated = Signal(str)

    def __init__(self):
        super().__init__()
        self.node = None
        self.timer = None
        self.aruco_pose_pub = None
        self.person_detection_pub = None
        self.state_log_pattern = re.compile(r'^[A-Z_]+ 상태 (진입|탈출)$')

    def run(self):
        rclpy.init()
        self.node = Node("pickee_dashboard_node")
        
        # ArucoPose publisher 생성
        self.aruco_pose_pub = self.node.create_publisher(
            ArucoPose,
            '/pickee/mobile/aruco_pose',
            10
        )

        # PersonDetection publisher 생성
        self.person_detection_pub = self.node.create_publisher(
            PersonDetection,
            '/pickee/mobile/person_detection',
            10
        )

        # /rosout 토픽을 구독하여 모든 로그 메시지 수신
        self.node.create_subscription(
            Log,
            '/rosout',
            self.log_callback,
            10
        )

        # /pickee/robot_status 토픽을 구독하여 상태 정보 수신
        self.node.create_subscription(
            PickeeRobotStatus,
            '/pickee/robot_status',
            self.status_callback,
            10
        )

        # 1초마다 시스템 정보를 폴링하는 타이머 설정
        self.timer = self.node.create_timer(1.0, self.poll_system_info)

        rclpy.spin(self.node)
        
        # 스레드 종료 시 노드 정리
        self.node.destroy_node()
        rclpy.shutdown()

    def publish_aruco_pose(self, marker_id, tvec, rvec):
        """아르코 마커 포즈 발행"""
        if self.aruco_pose_pub is None:
            return

        msg = ArucoPose()
        msg.aruco_id = marker_id
        msg.x = float(tvec[0])
        msg.y = float(tvec[1])
        msg.z = float(tvec[2])
        msg.roll = float(rvec[0])
        msg.pitch = float(rvec[1])
        msg.yaw = float(rvec[2])

        self.aruco_pose_pub.publish(msg)

    def publish_person_detection(self, x1, y1, x2, y2, confidence, direction):
        if self.person_detection_pub is None:
            return
        
        msg = PersonDetection()
        msg.robot_id = 1  
        msg.direction = direction

        self.person_detection_pub.publish(msg)

    def log_callback(self, msg: Log):
        # 로그 레벨을 문자로 변환 (예: 10 -> DEBUG, 20 -> INFO)
        level_map = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}
        level = level_map.get(msg.level, 'UNKNOWN')
        
        is_state_log = bool(self.state_log_pattern.match(msg.msg))
        # 시그널을 통해 UI 스레드로 데이터 전달
        self.log_received.emit(msg.name, level, msg.msg, is_state_log)

    def status_callback(self, msg: PickeeRobotStatus):
        self.state_updated.emit(msg.state)

    def poll_system_info(self):
        if not self.node: return

        # 노드 목록 가져오기
        node_names = self.node.get_node_names_and_namespaces()
        filtered_nodes = [f"{ns}{n}" for n, ns in node_names if 'pick' in n or 'shop' in n or 'mock' in n]
        self.nodes_updated.emit(filtered_nodes)

        # 토픽 목록 가져오기
        topic_names_and_types = self.node.get_topic_names_and_types()
        filtered_topics = [(name, types) for name, types in topic_names_and_types if 'pick' in name or 'shop' in name or 'mock' in name]
        self.topics_updated.emit(filtered_topics)

        # 서비스 목록 가져오기
        service_names_and_types = self.node.get_service_names_and_types()
        filtered_services = [
            (name, types) for name, types in service_names_and_types 
            if ('pick' in name or 'shop' in name or 'mock' in name) and not any(t.startswith('rcl_interfaces') or t.startswith('type_description_interfaces') for t in types)
        ]
        self.services_updated.emit(filtered_services)

    def stop(self):
        if self.node and rclpy.ok():
            rclpy.shutdown()
        self.wait() # 스레드가 완전히 종료될 때까지 대기
