#!/usr/bin/env python3
"""Packee Main과 myCobot 280 하드웨어를 연결하는 ROS2 서비스 노드."""


import copy
import threading
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from shopee_interfaces.msg import ArmPoseStatus
from shopee_interfaces.msg import ArmTaskStatus
from shopee_interfaces.msg import Pose6D
from shopee_interfaces.srv import ArmMoveToPose
from shopee_interfaces.srv import ArmPickProduct
from shopee_interfaces.srv import ArmPlaceProduct

try:
    from pymycobot.mycobot280 import MyCobot280
except ImportError:  # pragma: no cover - 하드웨어 라이브러리를 설치하지 않은 환경에서도 노드를 로드 가능하게 유지
    MyCobot280 = None


class PymycobotRightArmNode(Node):
    """Packee Main에서 내려오는 서비스를 처리하고 우측 myCobot 280을 직접 구동한다."""

    def __init__(self) -> None:
        super().__init__('pymycobot_right_arm_node')
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 1000000)
        self.declare_parameter('move_speed', 30)
        self.declare_parameter('arm_sides', 'right')
        self.declare_parameter('approach_offset_m', 0.02)
        self.declare_parameter('lift_offset_m', 0.03)
        self.declare_parameter(
            'preset_pose_cart_view',
            [142.0, -23.5, 291.5, -174.67, 5.73, -93.55])
        self.declare_parameter(
            'preset_pose_standby',
            [57.6, -63.4, 407.7, -93.33, 0.83, -88.72])
        self.declare_parameter('gripper_open_value', 100)
        self.declare_parameter('gripper_close_value', 0)
        self.declare_parameter('pose_status_topic', '/packee/arm/pose_status')
        self.declare_parameter('pick_status_topic', '/packee/arm/pick_status')
        self.declare_parameter('place_status_topic', '/packee/arm/place_status')

        self._serial_port: str = str(self.get_parameter('serial_port').value)
        self._baud_rate: int = int(self.get_parameter('baud_rate').value)
        self._move_speed: int = int(self.get_parameter('move_speed').value)
        sides_param = str(self.get_parameter('arm_sides').value)
        self._arm_sides: List[str] = [
            side.strip() for side in sides_param.split(',') if side.strip()
        ] or ['right']
        self._approach_offset: float = float(self.get_parameter('approach_offset_m').value)
        self._lift_offset: float = float(self.get_parameter('lift_offset_m').value)
        self._gripper_open_value: int = int(self.get_parameter('gripper_open_value').value)
        self._gripper_close_value: int = int(self.get_parameter('gripper_close_value').value)

        self._pose_presets = {
            'cart_view': self._parse_pose_parameter('preset_pose_cart_view'),
            'standby': self._parse_pose_parameter('preset_pose_standby')
        }
        self._pose_aliases = {'ready_pose': 'cart_view'}

        pose_status_topic = str(self.get_parameter('pose_status_topic').value)
        pick_status_topic = str(self.get_parameter('pick_status_topic').value)
        place_status_topic = str(self.get_parameter('place_status_topic').value)

        self._pose_status_pub = self.create_publisher(ArmPoseStatus, pose_status_topic, 10)
        self._pick_status_pub = self.create_publisher(ArmTaskStatus, pick_status_topic, 10)
        self._place_status_pub = self.create_publisher(ArmTaskStatus, place_status_topic, 10)

        self._move_service = self.create_service(
            ArmMoveToPose,
            '/packee1/arm/move_to_pose',
            self._handle_move_to_pose)
        self._pick_service = self.create_service(
            ArmPickProduct,
            '/packee1/arm/pick_product',
            self._handle_pick_product)
        self._place_service = self.create_service(
            ArmPlaceProduct,
            '/packee1/arm/place_product',
            self._handle_place_product)

        self._lock = threading.Lock()
        self._robot: Optional[MyCobot280] = None
        self._holding_product: Dict[str, Optional[int]] = {side: None for side in self._arm_sides}
        self._last_pose: Dict[str, Dict[str, float]] = {
            side: copy.deepcopy(self._pose_presets['standby']) for side in self._arm_sides
        }

        self._connect_robot()

        self.get_logger().info('우측 팔용 pymycobot 기반 Packee Arm 서비스 노드를 시작했습니다.')

    def _parse_pose_parameter(self, name: str) -> Dict[str, float]:
        """런치 파라미터에 정의된 포즈 배열을 사전으로 변환한다."""
        raw_value = self.get_parameter(name).value
        if isinstance(raw_value, (list, tuple)):
            values = list(raw_value)
        elif isinstance(raw_value, str):
            stripped = raw_value.strip().lstrip('[').rstrip(']')
            if stripped:
                try:
                    values = [float(item.strip()) for item in stripped.split(',')]
                except ValueError:
                    values = []
            else:
                values = []
        else:
            values = []
        if len(values) != 6:
            self.get_logger().warn(f'{name} 파라미터가 6개 항목을 갖지 않아 기본값을 사용합니다.')
            if 'cart_view' in name:
                return {'x': 0.16, 'y': 0.0, 'z': 0.18, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0}
            return {'x': 0.10, 'y': 0.0, 'z': 0.14, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0}
        return {
            'x': float(values[0]),
            'y': float(values[1]),
            'z': float(values[2]),
            'rx': float(values[3]),
            'ry': float(values[4]),
            'rz': float(values[5])
        }

    def _connect_robot(self) -> None:
        """Mycobot 280 시리얼 포트에 연결한다."""
        if MyCobot280 is None:
            self.get_logger().error('pymycobot 패키지를 찾을 수 없어 하드웨어 제어를 비활성화합니다.')
            return
        try:
            self._robot = MyCobot280(self._serial_port, self._baud_rate)
            if self._robot.is_power_on() != 1:
                self._robot.power_on()
            self._robot.focus_all_servos()
            connection_msg = (
                f'myCobot 280 연결 성공: 포트={self._serial_port}, '
                f'속도={self._baud_rate}')
            self.get_logger().info(connection_msg)
        except Exception as exc:  # pragma: no cover - 실제 하드웨어 환경에서만 평가
            self._robot = None
            self.get_logger().error(f'myCobot 280 연결 실패: {exc}')

    def _handle_move_to_pose(
        self,
        request: ArmMoveToPose.Request,
        response: ArmMoveToPose.Response
    ) -> ArmMoveToPose.Response:
        """Packee Main에서 내려온 자세 변경 명령을 처리한다."""
        pose_type = self._normalize_pose_type(request.pose_type)
        if pose_type not in self._pose_presets:
            response.success = False
            response.message = f'미지원 pose_type: {request.pose_type}'
            self.get_logger().warn(response.message)
            return response
        if not self._robot:
            response.success = False
            response.message = 'myCobot 연결이 활성화되지 않았습니다.'
            self.get_logger().error(response.message)
            return response

        target_pose = copy.deepcopy(self._pose_presets[pose_type])
        self._publish_pose_status(
            request.robot_id,
            request.order_id,
            pose_type,
            'in_progress',
            0.1,
            '자세 이동 명령을 수락했습니다.')
        with self._lock:
            if not self._send_pose(target_pose):
                response.success = False
                response.message = '자세 이동 중 오류가 발생했습니다.'
                self._publish_pose_status(
                    request.robot_id,
                    request.order_id,
                    pose_type,
                    'failed',
                    0.0,
                    response.message)
                return response
        for side in self._arm_sides:
            self._last_pose[side] = copy.deepcopy(target_pose)
        self._publish_pose_status(
            request.robot_id,
            request.order_id,
            pose_type,
            'in_progress',
            0.6,
            '목표 자세로 이동 중입니다.')
        self._sleep(1.0)
        self._publish_pose_status(
            request.robot_id,
            request.order_id,
            pose_type,
            'complete',
            1.0,
            '자세 이동을 완료했습니다.')
        response.success = True
        response.message = '자세 변경 명령을 완료했습니다.'
        return response

    def _handle_pick_product(
        self,
        request: ArmPickProduct.Request,
        response: ArmPickProduct.Response
    ) -> ArmPickProduct.Response:
        """Packee Main의 픽업 명령을 수행한다."""
        arm_side = request.arm_side or 'right'
        if arm_side not in self._arm_sides:
            response.success = False
            response.message = f'팔 정보가 올바르지 않습니다: {arm_side}'
            self.get_logger().warn(response.message)
            return response
        if not self._robot:
            response.success = False
            response.message = 'myCobot 연결이 활성화되지 않았습니다.'
            self.get_logger().error(response.message)
            return response

        pose = self._pose_from_msg(request.pose)
        self._publish_pick_status(
            request.robot_id,
            request.order_id,
            request.product_id,
            arm_side,
            'in_progress',
            'planning',
            0.05,
            '픽업 경로를 계획합니다.')
        approach_pose = copy.deepcopy(pose)
        approach_pose['z'] += self._approach_offset

        with self._lock:
            if not self._execute_pick_sequence(
                request.robot_id,
                request.order_id,
                request.product_id,
                arm_side,
                approach_pose,
                pose
            ):
                response.success = False
                response.message = '픽업 시퀀스가 실패했습니다.'
                self._publish_pick_status(
                    request.robot_id,
                    request.order_id,
                    request.product_id,
                    arm_side,
                    'failed',
                    'approaching',
                    0.4,
                    response.message)
                return response

        self._holding_product[arm_side] = request.product_id
        self._publish_pick_status(
            request.robot_id,
            request.order_id,
            request.product_id,
            arm_side,
            'completed',
            'done',
            1.0,
            '상품 픽업을 완료했습니다.')
        response.success = True
        response.message = '픽업 명령을 완료했습니다.'
        return response

    def _handle_place_product(
        self,
        request: ArmPlaceProduct.Request,
        response: ArmPlaceProduct.Response
    ) -> ArmPlaceProduct.Response:
        """Packee Main의 상품 담기 명령을 수행한다."""
        arm_side = request.arm_side or 'right'
        if arm_side not in self._arm_sides:
            response.success = False
            response.message = f'팔 정보가 올바르지 않습니다: {arm_side}'
            self.get_logger().warn(response.message)
            return response
        if not self._robot:
            response.success = False
            response.message = 'myCobot 연결이 활성화되지 않았습니다.'
            self.get_logger().error(response.message)
            return response
        if self._holding_product.get(arm_side) is None:
            response.success = False
            response.message = '해당 팔이 상품을 보유하고 있지 않습니다.'
            self.get_logger().warn(response.message)
            return response

        pose = self._pose_from_msg(request.pose)
        approach_pose = copy.deepcopy(pose)
        approach_pose['z'] += self._approach_offset

        self._publish_place_status(
            request.robot_id,
            request.order_id,
            request.product_id,
            arm_side,
            'in_progress',
            'planning',
            0.05,
            '담기 경로를 계획합니다.')
        with self._lock:
            if not self._execute_place_sequence(
                request.robot_id,
                request.order_id,
                request.product_id,
                arm_side,
                approach_pose,
                pose
            ):
                response.success = False
                response.message = '담기 시퀀스가 실패했습니다.'
                self._publish_place_status(
                    request.robot_id,
                    request.order_id,
                    request.product_id,
                    arm_side,
                    'failed',
                    'approaching',
                    0.4,
                    response.message)
                return response

        self._holding_product[arm_side] = None
        self._publish_place_status(
            request.robot_id,
            request.order_id,
            request.product_id,
            arm_side,
            'completed',
            'done',
            1.0,
            '상품 담기를 완료했습니다.')
        response.success = True
        response.message = '담기 명령을 완료했습니다.'
        return response

    def _execute_pick_sequence(
        self,
        robot_id: int,
        order_id: int,
        product_id: int,
        arm_side: str,
        approach_pose: Dict[str, float],
        grasp_pose: Dict[str, float]
    ) -> bool:
        """접근 → 하강 → 파지 → 상승 순으로 픽업을 수행한다."""
        self._publish_pick_status(
            robot_id,
            order_id,
            product_id,
            arm_side,
            'in_progress',
            'approaching',
            0.2,
            '목표 위치로 접근합니다.')
        if not self._send_pose(approach_pose):
            return False
        self._sleep(0.5)
        self._publish_pick_status(
            robot_id,
            order_id,
            product_id,
            arm_side,
            'in_progress',
            'approaching',
            0.4,
            '상품 위로 이동했습니다.')
        if not self._send_pose(grasp_pose):
            return False
        self._sleep(0.5)
        self._publish_pick_status(
            robot_id,
            order_id,
            product_id,
            arm_side,
            'in_progress',
            'grasping',
            0.6,
            '그리퍼를 닫습니다.')
        if not self._close_gripper():
            return False
        self._sleep(0.5)
        lift_pose = copy.deepcopy(grasp_pose)
        lift_pose['z'] += self._lift_offset
        if not self._send_pose(lift_pose):
            return False
        self._sleep(0.5)
        self._publish_pick_status(
            robot_id,
            order_id,
            product_id,
            arm_side,
            'in_progress',
            'lifting',
            0.8,
            '상품을 들어올렸습니다.')
        self._last_pose[arm_side] = copy.deepcopy(lift_pose)
        return True

    def _execute_place_sequence(
        self,
        robot_id: int,
        order_id: int,
        product_id: int,
        arm_side: str,
        approach_pose: Dict[str, float],
        place_pose: Dict[str, float]
    ) -> bool:
        """접근 → 하강 → 개방 → 상승 순으로 담기를 수행한다."""
        self._publish_place_status(
            robot_id,
            order_id,
            product_id,
            arm_side,
            'in_progress',
            'approaching',
            0.2,
            '포장 위치로 접근합니다.')
        if not self._send_pose(approach_pose):
            return False
        self._sleep(0.5)
        self._publish_place_status(
            robot_id,
            order_id,
            product_id,
            arm_side,
            'in_progress',
            'approaching',
            0.4,
            '포장 위치 위로 이동했습니다.')
        if not self._send_pose(place_pose):
            return False
        self._sleep(0.5)
        self._publish_place_status(
            robot_id,
            order_id,
            product_id,
            arm_side,
            'in_progress',
            'moving',
            0.6,
            '상품을 내려놓습니다.')
        if not self._open_gripper():
            return False
        self._sleep(0.5)
        lift_pose = copy.deepcopy(place_pose)
        lift_pose['z'] += self._lift_offset
        if not self._send_pose(lift_pose):
            return False
        self._last_pose[arm_side] = copy.deepcopy(lift_pose)
        self._sleep(0.5)
        return True

    def _normalize_pose_type(self, pose_type: str) -> str:
        """pose_type 별칭을 표준 pose 이름으로 변환한다."""
        key = pose_type.strip().lower()
        if key in self._pose_presets:
            return key
        return self._pose_aliases.get(key, key)

    def _pose_from_msg(self, pose: Pose6D) -> Dict[str, float]:
        """Pose6D 메시지를 내부 표현으로 변환한다."""
        return {
            'x': float(pose.x),
            'y': float(pose.y),
            'z': float(pose.z),
            'rx': float(pose.rx),
            'ry': float(pose.ry),
            'rz': float(pose.rz)
        }

    def _send_pose(self, pose: Dict[str, float]) -> bool:
        """Mycobot 280에 좌표 명령을 전송한다."""
        if not self._robot:
            return False
        coords_mm = [
            pose['x'],
            pose['y'],
            pose['z'],
            pose['rx'],
            pose['ry'],
            pose['rz'],
        ]
        try:
            self._robot.sync_send_coords(coords_mm, self._move_speed, 0)
            return True
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'좌표 전송 실패: {exc}')
            return False

    def _close_gripper(self) -> bool:
        """그리퍼를 닫는다."""
        if not self._robot:
            return False
        try:
            self._robot.set_gripper_value(self._gripper_close_value, max(10, self._move_speed))
            return True
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'그리퍼 닫기 실패: {exc}')
            return False

    def _open_gripper(self) -> bool:
        """그리퍼를 연다."""
        if not self._robot:
            return False
        try:
            self._robot.set_gripper_value(self._gripper_open_value, max(10, self._move_speed))
            return True
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'그리퍼 열기 실패: {exc}')
            return False

    def _publish_pose_status(
        self,
        robot_id: int,
        order_id: int,
        pose_type: str,
        status: str,
        progress: float,
        message: str
    ) -> None:
        """Pose 상태 토픽을 발행한다."""
        msg = ArmPoseStatus()
        msg.robot_id = robot_id
        msg.order_id = order_id
        msg.pose_type = pose_type
        msg.status = status
        msg.progress = progress
        msg.message = message
        self._pose_status_pub.publish(msg)

    def _publish_pick_status(
        self,
        robot_id: int,
        order_id: int,
        product_id: int,
        arm_side: str,
        status: str,
        phase: str,
        progress: float,
        message: str
    ) -> None:
        """픽업 상태 토픽을 발행한다."""
        msg = ArmTaskStatus()
        msg.robot_id = robot_id
        msg.order_id = order_id
        msg.product_id = product_id
        msg.status = status
        msg.current_phase = phase
        msg.progress = progress
        msg.message = message
        msg.arm_side = arm_side
        self._pick_status_pub.publish(msg)

    def _publish_place_status(
        self,
        robot_id: int,
        order_id: int,
        product_id: int,
        arm_side: str,
        status: str,
        phase: str,
        progress: float,
        message: str
    ) -> None:
        """담기 상태 토픽을 발행한다."""
        msg = ArmTaskStatus()
        msg.robot_id = robot_id
        msg.order_id = order_id
        msg.product_id = product_id
        msg.status = status
        msg.current_phase = phase
        msg.progress = progress
        msg.message = message
        msg.arm_side = arm_side
        self._place_status_pub.publish(msg)

    @staticmethod
    def _sleep(duration: float) -> None:
        """서비스 처리 중 내부 대기 시간을 명확히 표현한다."""
        time.sleep(duration)


def main() -> None:
    rclpy.init()
    node = PymycobotRightArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('pymycobot 노드를 종료합니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
