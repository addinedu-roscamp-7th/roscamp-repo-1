#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 수학적 핸들링을 위한 math 라이브러리 import
import math
# 시간 handle을 위한 time 라이브러리 import
import time
# rclpy 라이브러리 import
import rclpy
# 노드 생성을 위한 Node2 라이브러리 추가
from rclpy.node import Node
# 퍼블리셔, 서브스크라이버 서비스 품질을 위한 QOS 설정을 위해 QoSProfile import
from rclpy.qos import QoSProfile
# QOS 신뢰성 정책(RELIABLE/BEST_EFFORT) 사용을 위한 QoSReliabilityPolicy import
from rclpy.qos import QoSReliabilityPolicy
# QoS 내구성 정책(TRANSIENT_LOCAL/VOLATILE) 사용을 위한 QoSDurabilityPolicy 임포트
from rclpy.qos import QoSDurabilityPolicy
# 속도 명령을 위한 Twist 라이브러리 import
from geometry_msgs.msg import Twist
# 오도메트리 구독을 위한 Odometry 라이브러리 import
from nav_msgs.msg import Odometry

# OdomMove Class 선언
class OdomMove(Node):
    """ 현재 오도메트리 정보를 받아와서 현재 위치로 활용 후 
        이동할 거리를 입력하면 해당 거리만큼 속도 명령을 발행하여 주행하는 간단한 주행 코드 """

    # 클래스 초기화
    # 클래스 인자 : self
    # 움직이고 싶은 거리 : target_distance
    def __init__(self, target_distance: float):

        # 노드 이름 언선
        super().__init__('odom_move')

        # ---- 파라미터 선언 ----

        # odom_topic을 /odom 값으로 설정
        # /odom : 자기 위치 추정 정보를 담고 있는 토픽
        self.declare_parameter('odom_topic', '/odom')
        # vel_topic을 /cmd_vel 값으로 설정
        # /cmd_vel : 직선 속도와 각속도를 제어할수 있는 토픽
        self.declare_parameter('vel_topic', '/cmd_vel_modified')
        # speed_linear를 1.45 m/s 로 설정
        # speed_linear : 직선 속도
        self.declare_parameter('speed_linear', 0.20)
        # speed_angular를 00.0 rad/s 로 설정
        # speed_angular : 각속도
        self.declare_parameter('speed_angular', 0.0)
        # tolerance를 0.05m(50cm)로 설정
        self.declare_parameter('tolerance', 0.05)
        # rate_hz를 20hz로 설정
        # rate_hz : 제어 주기
        self.declare_parameter('rate_hz', 20)
        

        # ---- 파라미터 로드 ----
        # odom_topic 변수에 odom_topic 값 저장
        self.odom_topic = self.get_parameter('odom_topic').value
        # vel_topic 변수에 vel_topic 값 저장
        self.vel_topic  = self.get_parameter('vel_topic').value
        # target 변수에 target_distance 값 저장
        self.target     = float(target_distance)
        # vx_mag 변수에 선속도 절대값 저장
        self.vx_mag     = abs(float(self.get_parameter('speed_linear').value))
        # tol 변수에 tolerance 값 저장
        self.tol        = float(self.get_parameter('tolerance').value)
        # rate_hz 변수에 rate_hz 값 저장
        self.rate_hz    = max(5, int(self.get_parameter('rate_hz').value))
        # wz 변수에 각속도 값 저장
        self.wz         = float(self.get_parameter('speed_angular').value)

        # ---- 통신 ----
        # 오도메트리 관련 QOS 프로파일 설정
        # depth : 명령을 몇개까지 유지할 것인지
        # reliability : 패킷이 손실되면 재전송하는등 최대한 전달 보장을 지정(RELIABLE) / 조금 빠져도 속도가 중요함(BEST_EFFORT)
        # durability : 퍼블리셔가 송신한 메세지를 저장하지 않음, 구독자가 늘러놔도 과거 메시지를 송출하지 x(VOLATILE)
        odom_qos = QoSProfile(depth=10,
                              reliability=QoSReliabilityPolicy.BEST_EFFORT,
                              durability=QoSDurabilityPolicy.VOLATILE)
        # create_subscription(메세지 형식, 구독할 토픽 이름, 콜백 함수, QOS 정책) : 어떤 토픽을 구독할지 설정하는 함수
        # 콜백 함수 : 토픽 구독이 되면 실행되는 함수
        # Odometry 메시지 형식의 odom_topic에 저장된 토픽을 구독하고 구독이 되면 _on_odom 이 실행
        self.sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, odom_qos)

        # 속도 관련 QOS 프로파일 설정
        vel_qos = QoSProfile(depth=10,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.VOLATILE)
        
        # create_publisher(메시지 형식, 토픽 이름, QOS 정책)
        # Twist 형식으로 vel_topic에 저장된 topic 이름으로 토픽 발행
        self.pub = self.create_publisher(Twist, self.vel_topic, vel_qos)

        # ---- 상태 ----
        # 시작점 좌표 값 x0, y0 None으로 설정
        self.x0 = self.y0 = None
        # 현재 좌표 x,y None으로 설정
        self.x = self.y = None
        # 오도메트리 수신 시간 지금 시간으로 설정
        self.last_odom_time = time.time()
        # 주행 시작 시간 None으로 설정ㄴ
        self.start_time = None
        # 예상 주행 시간의 3~5배를 timeout 시간으로 설정
        self.timeout = max(5.0, abs(self.target)/max(self.vx_mag, 0.01)*3.0)
        # 주행 완료 여부 Flag Fasle로 설정
        # True : 주행 완료
        # False : 주행 미완료
        self.done = False

        # ---- 타이머 ----
        # rate_hz 간격으로 _step 함수 실행 
        self.timer = self.create_timer(1.0/self.rate_hz, self._step)

        # 초기 상태 로그 출력
        self.get_logger().info(
            f'현재 위치={self.odom_topic}, 이동 거리 ={self.target:.2f}m, '
            f'선속도={self.vx_mag:.2f}m/s, 허용 오차={self.tol:.2f}m, 제어 주기={self.rate_hz}Hz, '
            f'timeout≈{self.timeout:.1f}s'
        )

    # 오도메트리 콜백 함수 선언
    def _on_odom(self, msg: Odometry):
        # 현재 위치를 변수 x에 저장
        self.x = msg.pose.pose.position.x
        # 현재 위치를 변수 y에 저장
        self.y = msg.pose.pose.position.y
        # 오도메트리 수신 시간을 현재 시간 저장
        self.last_odom_time = time.time()
        # 초기 위치를 모르면
        if self.x0 is None:
            # 초기 위치를 현재 오도매트리 값으로 저장
            self.x0, self.y0 = self.x, self.y
            # 주행 시작 시간을 현재 오도메트리가 저장된 시간으로 설정
            self.start_time = self.last_odom_time
            # 시작 좌표를 오도메트리 좌표로 log 출력
            # log : ros에서 print 같은거
            self.get_logger().info(f'start @ ({self.x0:.3f}, {self.y0:.3f})')

    # 목표에 도달하였으면
    def _step(self):
        # 주행이 완료가 되면 아무것도 안함
        if self.done:
            return
        
        # 현재 시각 now에 저장
        now = time.time()

        # 시작 위치가 정해지지 않았다면
        if self.x0 is None:
            # 오도메트리 수신 이후로 1.5초가 지났으면
            if now - self.last_odom_time > 1.5:
                # aiting /odom ...' log 출력
                self.get_logger().warn('waiting /odom ...')
            return

        # 오도메트리가 1.0초 동안 수신이 안되면
        if now - self.last_odom_time > 1.0:
            # 오도메트리 수신이 안되서 정지 로그 출력
            self.get_logger().error('odom lost → stop')
            # 코드 종료
            self._finish()
            return

        # 초기 위치에서 현재 위치까지 이동거리 계산하여 moved에 저장
        moved = math.hypot(self.x - self.x0, self.y - self.y0)
        # 이동한 거리 목적지 거리까지 거리가 tolerance 보다 작거나 같으면
        if abs(self.target) - moved <= self.tol:
            # 도착했다는 로그 출력
            self.get_logger().info(f'reached: moved={moved:.3f}m')
            # 코드 종료
            self._finish()
            return

        # timeout 초과되었다면
        if now - self.start_time > self.timeout:
            # timeout log 출력
            self.get_logger().warn(f'timeout {self.timeout:.1f}s')
            # 코드 출력
            self._finish()
            return

        # 속도 제어 객체 생성
        cmd = Twist()
        # 타겟이 양수면 + 선속도, 타겟이 음수면 -선속도 설정
        cmd.linear.x = self.vx_mag if self.target >= 0 else -self.vx_mag
        # 속도 명령 퍼플리시
        self.pub.publish(cmd)
    
    # 종료 함수 선언
    def _finish(self):
        # 정지 객체 생성
        stop = Twist()
        # stop 퍼플리시
        self.pub.publish(stop)
        # 안전을 위해 한번 더 퍼플리시
        self.pub.publish(stop)
        # STOP log 출력
        self.get_logger().info('STOP')
        # 종료 플래그 True로 설정
        self.done = True


# target_distance 인자로 받아서 주행하는 함수 선언
# target_distance : 움직이고 싶은 거리
def run(target_distance: float):
    # rclpy 초기화
    rclpy.init()
    # OdomMove 클래스 객체 생성
    node = OdomMove(target_distance)
    # 에러가 없으면
    try:
        # 이상이 없을 때 까지
        while rclpy.ok() and not node.done:
            # 노드 0.1초 단위로 반복
            rclpy.spin_once(node, timeout_sec=0.1)
    # 키보드 인터럽트 발생 시 ctrl+c
    except KeyboardInterrupt:
        # 노드 종료
        node._finish()
        # ros2 종료
        rclpy.shutdown()
    # 최종적으로
    finally:

        try:
            # 노드 제거
            node.destroy_node()
        finally:
            # rclpy 상태가 정상적이라면
            if rclpy.ok():
                # rclpy 종료
                rclpy.shutdown()


def main():
    # 앞으로 1.5m 이동
    run(0.5)
    # 뒤로 1.5m 이동
    run(-0.5)
    # 앞으로 0.8m 이동
    run(0.8)
    #  뒤로 0.5m  이동
    run(-0.5)

if __name__ == '__main__':
    # 메인 함수 실행
    main()