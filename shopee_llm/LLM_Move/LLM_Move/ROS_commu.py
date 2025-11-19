# --------------------------- ROS2 관련 라이브러리 선언 ---------------------
# rclpy import
import rclpy
# 퍼블리셔, 서브스크라이버 서비스 품질을 위한 QOS 설정을 위해 QoSProfile import
from rclpy.qos import QoSProfile
# QOS 신뢰성 정책(RELIABLE/BEST_EFFORT) 사용을 위한 QoSReliabilityPolicy import
from rclpy.qos import QoSReliabilityPolicy
# QoS 내구성 정책(TRANSIENT_LOCAL/VOLATILE) 사용을 위한 QoSDurabilityPolicy 임포트
from rclpy.qos import QoSDurabilityPolicy
# 노드 생성을 위한 Node 라이브러리 import
from rclpy.node import Node
# Float32 형식으로 메시지를 발행하기 위해 Float32 라이브러리 import
from std_msgs.msg import Float32
# String 형식으로 메시지를 발행하기 위해 Float32 라이브러리 import
from std_msgs.msg import String
# 서비스 타입 import
from shopee_interfaces.srv import ChangeTrackingMode

# ROS 유틸함수 선언을 위해 ROS Util class 선언 
class ROS_Util():
    # 초기화 함수 선언
    def __init__(self):
        # 노드 이름 설정
        self.node_name = "topic_publisher"
        # 거리 기반 이동 토픽 이름 설정
        self.distance_topic_name = "/move_distance"
        # 거리 기반 이동 토픽 이름 설정
        self.place_topic_name = "/move_place"
        # 팔로잉 서비스 이름 설정
        self.change_mode_service_name = "/pickee/mobile/change_tracking_mode"
        # rclpy가 초기화 안되었으면 초기화
        rclpy.init()
        # 노드 생성
        self.node = Node(self.node_name )
        # qos 설정
        # QOS 프로파일 설정
        # depth : 명령을 몇개까지 유지할 것인지
        # reliability : 패킷이 손실되면 재전송하는등 최대한 전달 보장을 지정(RELIABLE) / 조금 빠져도 속도가 중요함(BEST_EFFORT)
        # durability : 퍼블리셔가 송신한 메세지를 저장하지 않음, 구독자가 늘러놔도 과거 메시지를 송출하지 x(VOLATILE)
        self.qos = QoSProfile(depth=10,
                     reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.VOLATILE)
        # float 퍼블리셔 생성
        self.float_publisher = self.node.create_publisher(Float32,self.distance_topic_name,self.qos)
        # string 퍼블리셔 생성
        self.string_publisher = self.node.create_publisher(String,self.place_topic_name,self.qos)
        # 서비스 클라이언트 생성
        self.change_mode_client = self.node.create_client(ChangeTrackingMode,self.change_mode_service_name)

    # float32 메시지 퍼블리시 함수 선언
    def publish_float32(self,usr_data):
        # 에러가 없으면
        try:
            # Float32 형식의 메시지 생성
            msg = Float32()
            # 인자로 받은 데이터를 msg.data로 설정
            msg.data = float(usr_data)
            # 메시지 퍼블리시
            self.float_publisher.publish(msg)
            # log 출력
            self.node.get_logger().info(f"[ROS] /Float32 publish : {msg.data}")
        # 에러가 발생하면
        except Exception as e:
            # 에러 log 출력
            self.node.get_logger().error(f"[ROS] /Float32 publish 실패 : {e}")

    # string 메시지 퍼블리시 함수 선언 
    def publish_string(self,usr_data):
        # 에러가 없으면
        try:
            # Float32 형식의 메시지 생성
            msg = String()
            # 인자로 받은 데이터를 msg.data로 설정
            msg.data = str(usr_data)
            # 메시지 퍼블리시
            self.string_publisher.publish(msg)
            # log 출력
            self.node.get_logger().info(f"[ROS] String publish : {msg.data}")
        # 에러가 발생하면
        except Exception as e:
            # 에러 log 출력
            self.node.get_logger().error(f"[ROS] String publish 실패 : {e}")

    # 팔로잉 요청을 위해서 서비스 콜(클라이언트) 함수 선언
    def service_call(self,robot_id,mode,timeout_sec):
        # 에러가 없으면
        try:
            # 서비스 서버가 준비가 안되었는지 확인
            if not self.change_mode_client.wait_for_service(timeout_sec=timeout_sec):
                # timeout_sec동안 서버와 연결이 안되면 서비스 대기 시간 초과 로그 출력
                self.node.get_logger().error(f"[ROS] 서비스 대기 시간 초과: {self.change_mode_service_name}")
            # 서비스 콜 객체 선언
            service_request = ChangeTrackingMode.Request()
            # .srv에 미리 선언된 robot_id값을 robot_id 값으로 설정
            service_request.robot_id = int(robot_id)
            # .srv에 미리 선언된 mode값을 mode 값으로 설정
            service_request.mode = str(mode).strip()
            # 서비스 콜 요청 로그 출력
            self.node.get_logger().info(f"Service_Request : robot_id={service_request.robot_id}, mode='{service_request.mode}'")
            # 비동기 방식으로 서비스 콜 실행 (응답은 furture에 답겨서 나중에 도착)
            future = self.change_mode_client.call_async(service_request)
            # timeout_sec 동안 응답이 올 때까지 노드 이벤트 루프를 돌면서 대기
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
            # 응답이 수신되지 않았다면
            if not future.done():
                # 타임아웃 log 출력
                self.node.get_logger().warn(f"Timeout: no response within {self.timeout_sec} sec")
                # 다음 루프로 이동
                return
            # 응답 저장
            response = future.result()
                # 응답 출력
            self.node.get_logger().info(f"[SERVER REPLY] success={response.success}, message='{response.message}'")
        # 에러 발생 시
        except Exception as e:
                # 에러 로그 출력
                self.node.get_logger().error(f"Service call failed: {e}")

    # ros shutdown 관련 함수 선언
    def shutdown(self):
        # 에러가 없으면
        try:
            # float 퍼블리셔 종료
            self.float_publisher.destroy()
            # string 퍼블리셔 종료
            self.string_publisher.destroy()
        # 에러가 발생하면
        except Exception:
            # pass (아래에서 rclpy를 종료하기 때문)
            pass
        # 에러가 없으면
        try:
            # node 종료
            self.node.destroy_node()
        # 에러가 발생하면
        except Exception:
            # pass (아래에서 rclpy를 종료하기 때문)
            pass
        # rclpy 종료
        rclpy.shutdown()