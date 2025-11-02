#!/usr/bin/env python3
'''Packee Arm JetCobot 브릿지 노드.'''


import math
from typing import Dict, List, Optional, Sequence

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32

try:
    from pymycobot.mycobot280 import MyCobot280
except ImportError:  # pragma: no cover - 장비가 없는 환경에서도 노드를 기동 가능하게 유지
    MyCobot280 = None


class JetCobotArm:
    '''단일 myCobot 280 팔과 통신하며 속도/그리퍼 명령을 적용한다.'''

    def __init__(
        self,
        node: Node,
        arm_name: str,
        serial_port: str,
        move_speed: int,
        default_pose: Sequence[float],
        workspace: Dict[str, float],
        fallback_dt: float,
        gripper_open_value: int,
        gripper_close_value: int
    ) -> None:
        self._node = node
        self._arm_name = arm_name
        self._serial_port = serial_port
        self._move_speed = move_speed
        self._workspace = workspace
        self._fallback_dt = fallback_dt
        self._gripper_open_value = gripper_open_value
        self._gripper_close_value = gripper_close_value
        self._pose = {
            'x': float(default_pose[0]),
            'y': float(default_pose[1]),
            'z': float(default_pose[2]),
            'rx': float(default_pose[3]),
            'ry': float(default_pose[4]),
            'rz': float(default_pose[5])
        }
        self._last_stamp: Time = node.get_clock().now()
        self._robot: Optional[MyCobot280] = None
        self._connect()

    def _connect(self) -> None:
        '''시리얼 포트를 통해 JetCobot에 접속하고 초기화를 수행한다.'''
        if not self._serial_port:
            self._node.get_logger().warn(
                f'[{self._arm_name}] serial_port 파라미터가 비어 있어 실제 로봇 제어를 비활성화합니다.')
            return
        if MyCobot280 is None:
            self._node.get_logger().error(
                f'[{self._arm_name}] pymycobot 패키지를 찾을 수 없습니다. pip install pymycobot 후 다시 실행하세요.')
            return
        try:
            self._robot = MyCobot280(self._serial_port, 1000000)
            if self._robot.is_power_on() != 1:
                self._robot.power_on()
            self._robot.focus_all_servos()
            self._node.get_logger().info(
                f'[{self._arm_name}] 포트 {self._serial_port} 에 JetCobot 연결 및 서보 활성화를 완료했습니다.')
        except Exception as exc:  # pragma: no cover - 하드웨어 환경에서만 발생
            self._node.get_logger().error(f'[{self._arm_name}] JetCobot 연결에 실패했습니다: {exc}')
            self._robot = None

    def handle_twist(self, message: TwistStamped) -> None:
        '''Packee Arm 컨트롤러가 발행한 속도 명령을 적분해 좌표 이동을 수행한다.'''
        now_stamp = (
            Time.from_msg(message.header.stamp)
            if message.header.stamp.sec != 0
            else self._node.get_clock().now()
        )
        dt = max(0.02, min(0.3, (now_stamp - self._last_stamp).nanoseconds / 1e9))
        self._last_stamp = now_stamp
        if math.isnan(dt) or dt <= 0.0:
            dt = self._fallback_dt

        self._pose['x'] += message.twist.linear.x * dt
        self._pose['y'] += message.twist.linear.y * dt
        self._pose['z'] += message.twist.linear.z * dt
        self._pose['rz'] += message.twist.angular.z * dt

        radial = math.sqrt((self._pose['x'] ** 2) + (self._pose['y'] ** 2))
        if radial > self._workspace['radial']:
            scale = self._workspace['radial'] / max(radial, 1e-6)
            self._pose['x'] *= scale
            self._pose['y'] *= scale
        self._pose['z'] = min(self._workspace['z_max'], max(self._workspace['z_min'], self._pose['z']))

        if not self._robot:
            return

        coords_mm = [
            self._pose['x'] * 1000.0,
            self._pose['y'] * 1000.0,
            self._pose['z'] * 1000.0,
            math.degrees(self._pose['rx']),
            math.degrees(self._pose['ry']),
            math.degrees(self._pose['rz'])
        ]
        try:
            self._robot.sync_send_coords(coords_mm, self._move_speed, 0)
        except Exception as exc:  # pragma: no cover - 실제 장비에서만 발생
            self._node.get_logger().error(f'[{self._arm_name}] 좌표 명령 전송 실패: {exc}')

    def handle_gripper(self, message: Float32) -> None:
        '''GripperController가 발행한 힘 명령을 JetCobot gripper 값으로 변환한다.'''
        if not self._robot:
            return
        target_value = self._gripper_open_value if message.data <= 0.0 else self._gripper_close_value
        try:
            self._robot.set_gripper_value(target_value, max(10, self._move_speed))
        except Exception as exc:  # pragma: no cover
            self._node.get_logger().error(f'[{self._arm_name}] 그리퍼 명령 전송 실패: {exc}')


class JetCobotBridge(Node):
    '''Packee Arm ↔ JetCobot 하드웨어 통신을 담당하는 rclpy 노드.'''

    def __init__(self) -> None:
        super().__init__('jetcobot_bridge')
        self.declare_parameter('left_serial_port', '/dev/ttyUSB0')
        self.declare_parameter('right_serial_port', '/dev/ttyUSB1')
        self.declare_parameter('left_velocity_topic', '/packee/jetcobot/left/cmd_vel')
        self.declare_parameter('right_velocity_topic', '/packee/jetcobot/right/cmd_vel')
        self.declare_parameter('left_gripper_topic', '/packee/jetcobot/left/gripper_cmd')
        self.declare_parameter('right_gripper_topic', '/packee/jetcobot/right/gripper_cmd')
        self.declare_parameter('command_period_sec', 0.15)
        self.declare_parameter('move_speed', 40)
        self.declare_parameter('workspace_radial', 0.28)
        self.declare_parameter('workspace_z_min', 0.05)
        self.declare_parameter('workspace_z_max', 0.30)
        self.declare_parameter('default_pose_cart_view', [0.16, 0.0, 0.18, 0.0, 0.0, 0.0])
        self.declare_parameter('default_pose_standby', [0.10, 0.0, 0.14, 0.0, 0.0, 0.0])
        self.declare_parameter('gripper_open_value', 100)
        self.declare_parameter('gripper_close_value', 10)

        workspace = {
            'radial': float(self.get_parameter('workspace_radial').value),
            'z_min': float(self.get_parameter('workspace_z_min').value),
            'z_max': float(self.get_parameter('workspace_z_max').value)
        }
        move_speed = int(self.get_parameter('move_speed').value)
        fallback_dt = float(self.get_parameter('command_period_sec').value)
        open_value = int(self.get_parameter('gripper_open_value').value)
        close_value = int(self.get_parameter('gripper_close_value').value)

        def parse_pose_param(name: str, fallback: Sequence[float]) -> List[float]:
            '''파라미터 문자열/배열을 6자유도 포즈 리스트로 변환한다.'''
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
                return list(fallback)
            return [float(item) for item in values]

        cart_view_pose = parse_pose_param('default_pose_cart_view', [0.16, 0.0, 0.18, 0.0, 0.0, 0.0])
        standby_pose = parse_pose_param('default_pose_standby', [0.10, 0.0, 0.14, 0.0, 0.0, 0.0])

        self._left_arm = JetCobotArm(
            self,
            'left',
            self.get_parameter('left_serial_port').value,
            move_speed,
            cart_view_pose,
            workspace,
            fallback_dt,
            open_value,
            close_value
        )
        self._right_arm = JetCobotArm(
            self,
            'right',
            self.get_parameter('right_serial_port').value,
            move_speed,
            standby_pose,
            workspace,
            fallback_dt,
            open_value,
            close_value
        )

        self.create_subscription(
            TwistStamped,
            self.get_parameter('left_velocity_topic').value,
            self._left_arm.handle_twist,
            10
        )
        self.create_subscription(
            TwistStamped,
            self.get_parameter('right_velocity_topic').value,
            self._right_arm.handle_twist,
            10
        )
        self.create_subscription(
            Float32,
            self.get_parameter('left_gripper_topic').value,
            self._left_arm.handle_gripper,
            10
        )
        self.create_subscription(
            Float32,
            self.get_parameter('right_gripper_topic').value,
            self._right_arm.handle_gripper,
            10
        )

        self.get_logger().info('JetCobot 브릿지 노드를 기동했습니다.')


def main() -> None:
    rclpy.init()
    node = JetCobotBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('JetCobot 브릿지 노드를 종료합니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
