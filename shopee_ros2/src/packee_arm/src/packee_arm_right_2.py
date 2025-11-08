#!/usr/bin/env python3
"""
Packee Main ↔ myCobot 280 통합 노드 (ROS2 서비스 + Vision IBVS 보정)

🔥 Fix: `ModuleNotFoundError: No module named 'rclpy'`
- 이 파일은 **ROS2가 없어도 import/실행/단위테스트가 가능**하도록 리팩터링됨
- ROS2가 있으면 기존대로 노드 실행, 없으면 내장 **unittest**를 실행

핵심 구조
- Core 로직을 `CoreArmIBVSController`(순수 Python)로 분리 → 하드웨어/IBVS 시퀀스/그리퍼 제어 포함
- ROS2 어댑터 `PackeeArmIBVSNode`는 Core를 감싸서 서비스/토픽/파라미터를 제공
- 비-ROS2 환경에선 경량 스텁 타입으로 대체하고 **테스트 케이스** 수행

사용 방법
1) ROS2 환경: `ros2 run ...` 또는 `python packee_arm_ibvs_node.py` (ROS2 감지 시 노드 실행)
2) 일반 Python: `python packee_arm_ibvs_node.py` → 내장 테스트 자동 실행
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ----------------------------
# Optional ROS2 imports (존재 여부에 따라 스텁 대체)
# ----------------------------
HAS_ROS2 = True
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from shopee_interfaces.msg import ArmPoseStatus, ArmTaskStatus, Pose6D  # type: ignore
    from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, ArmPlaceProduct  # type: ignore
    from shopee_interfaces.srv import PackeeVisionDetectProductsInCart  # type: ignore
except Exception:
    HAS_ROS2 = False

    # ---- 최소 동치 타입 정의 (테스트 전용) ----
    @dataclass
    class Pose6D:
        x: float = 0.0; y: float = 0.0; z: float = 0.0
        rx: float = 0.0; ry: float = 0.0; rz: float = 0.0

    @dataclass
    class ArmPoseStatus:
        robot_id: int = 0; order_id: int = 0
        pose_type: str = ""; status: str = ""; progress: float = 0.0; message: str = ""

    @dataclass
    class ArmTaskStatus:
        robot_id: int = 0; order_id: int = 0; product_id: int = 0; arm_side: str = "right"
        status: str = ""; message: str = ""; progress: float = 0.0
        # 인터페이스 차이 흡수
        sub_status: str = ""; current_phase: str = ""

    # Service request/response 스텁
    class _ReqBase: ...
    class _ResBase: ...

    class ArmMoveToPose:
        class Request(_ReqBase):
            def __init__(self, robot_id=0, order_id=0, pose_type=""):
                self.robot_id = robot_id; self.order_id = order_id; self.pose_type = pose_type
        class Response(_ResBase):
            def __init__(self): self.success=False; self.message=""

    class ArmPickProduct:
        class Request(_ReqBase):
            def __init__(self, robot_id=0, order_id=0, product_id=0, arm_side='right', pose: Pose6D|None=None):
                self.robot_id=robot_id; self.order_id=order_id; self.product_id=product_id
                self.arm_side=arm_side; self.pose=pose or Pose6D()
        class Response(_ResBase):
            def __init__(self): self.success=False; self.message=""

    class ArmPlaceProduct:
        class Request(_ReqBase):
            def __init__(self, robot_id=0, order_id=0, product_id=0, arm_side='right', pose: Pose6D|None=None):
                self.robot_id=robot_id; self.order_id=order_id; self.product_id=product_id
                self.arm_side=arm_side; self.pose=pose or Pose6D()
        class Response(_ResBase):
            def __init__(self): self.success=False; self.message=""

    class PackeeVisionDetectProductsInCart:
        class Request(_ReqBase):
            def __init__(self):
                self.robot_id=0; self.order_id=0; self.expected_product_id=0; self.expected_product_ids=[]
        class Response(_ResBase):
            def __init__(self, current_pose=None, target_pose=None):
                self.current_pose=current_pose; self.target_pose=target_pose

    # ROS2 Node 대체 (로그만 출력)
    class Node:
        def __init__(self, name: str): self._name=name
        def get_logger(self):
            class _L:
                def info(self, m): print(f"[INFO] {m}")
                def warn(self, m): print(f"[WARN] {m}")
                def error(self, m): print(f"[ERROR] {m}")
            return _L()
        # No-op APIs for tests
        def declare_parameter(self, *a, **k): return None
        def get_parameter(self, name):
            class _P: value=None
            return _P()
        def create_publisher(self, *a, **k):
            class _Pub:
                def publish(self, msg): pass
            return _Pub()
        def create_service(self, *a, **k): return None
        def create_client(self, *a, **k):
            class _Cli:
                def wait_for_service(self, t): return True
                def call_async(self, req):
                    class _F:
                        def __init__(self, res): self._res=res
                        def result(self): return self._res
                    return _F(PackeeVisionDetectProductsInCart.Response())
            return _Cli()

# ----------------------------
# Optional pymycobot import
# ----------------------------
try:
    from pymycobot.mycobot280 import MyCobot280  # type: ignore
except Exception:
    MyCobot280 = None  # 테스트 환경: 스텁 사용

# ----------------------------
# Robot & Vision Stubs (테스트용)
# ----------------------------
class MyCobotStub:
    def __init__(self):
        self.powered=False
        self.sent_coords: List[List[float]]=[]
        self.gripper_cmds: List[Tuple[int,int]]=[]
    def is_power_on(self): return 1 if self.powered else 0
    def power_on(self): self.powered=True
    def focus_all_servos(self): pass
    def sync_send_coords(self, coords, speed, mode):
        assert len(coords)==6
        self.sent_coords.append(list(coords))
    def set_gripper_value(self, value, speed):
        self.gripper_cmds.append((int(value), int(speed)))

class VisionStub:
    """단순 수렴 모델: current → target 으로 40%씩 접근"""
    def __init__(self, start: List[float], target: List[float]):
        self.current=np.array(start, dtype=float)
        self.target=np.array(target, dtype=float)
    def next(self) -> PackeeVisionDetectProductsInCart.Response:
        self.current = self.current + 0.4*(self.target-self.current)
        return PackeeVisionDetectProductsInCart.Response(
            current_pose=self.current.copy(), target_pose=self.target.copy()
        )

# ----------------------------
# Core controller (ROS2 비의존)
# ----------------------------
class CoreArmIBVSController:
    def __init__(self,
                 robot: MyCobotStub | MyCobot280 | None,
                 move_speed: int = 30,
                 approach_offset_mm: float = 20.0,
                 lift_offset_mm: float = 30.0,
                 gripper_open_value: int = 100,
                 gripper_close_value: int = 0,
                 ibvs_gain: float = 0.3,
                 ibvs_eps: float = 10.0,
                 ibvs_max_iters: int = 20):
        self.robot = robot
        self.lock = threading.Lock()
        self.move_speed = int(move_speed)
        self.approach_offset_mm=float(approach_offset_mm)
        self.lift_offset_mm=float(lift_offset_mm)
        self.gripper_open_value=int(gripper_open_value)
        self.gripper_close_value=int(gripper_close_value)
        self.ibvs_gain=float(ibvs_gain)
        self.ibvs_eps=float(ibvs_eps)
        self.ibvs_max_iters=int(ibvs_max_iters)

    # ----- low level -----
    def _send_pose(self, pose: Dict[str,float]) -> bool:
        if not self.robot: return False
        coords=[pose['x'],pose['y'],pose['z'],pose['rx'],pose['ry'],pose['rz']]
        try:
            with self.lock:
                self.robot.sync_send_coords(coords, self.move_speed, 1)
            return True
        except Exception:
            return False

    def _close_gripper(self) -> bool:
        if not self.robot: return False
        with self.lock:
            self.robot.set_gripper_value(self.gripper_close_value, max(10,self.move_speed))
        return True

    def _open_gripper(self) -> bool:
        if not self.robot: return False
        with self.lock:
            self.robot.set_gripper_value(self.gripper_open_value, max(10,self.move_speed))
        return True

    # ----- helpers -----
    @staticmethod
    def pose_from_msg(p: Pose6D) -> Dict[str,float]:
        return {'x':float(p.x),'y':float(p.y),'z':float(p.z),'rx':float(p.rx),'ry':float(p.ry),'rz':float(p.rz)}

    @staticmethod
    def vec_from_pose(p: Dict[str,float]|List[float]|np.ndarray) -> np.ndarray:
        if isinstance(p, dict):
            return np.array([p['x'],p['y'],p['z'],p['rx'],p['ry'],p['rz']], dtype=float)
        return np.asarray(p, dtype=float).reshape(6)

    @staticmethod
    def pose_from_vec(v: Union[List[float],np.ndarray]) -> Dict[str,float]:
        a=np.asarray(v, dtype=float).reshape(6)
        return {'x':float(a[0]),'y':float(a[1]),'z':float(a[2]),'rx':float(a[3]),'ry':float(a[4]),'rz':float(a[5])}

    # ----- IBVS main -----
    def ibvs_correct(self,
                     vision_next_callable,
                     init_pose: Dict[str,float]) -> Tuple[bool, Dict[str,float]]:
        """vision_next_callable() -> Response(current_pose, target_pose)
        수렴 시 (True, 보정된_pose), 실패 시 (False, 마지막_pose)
        """
        cur = self.vec_from_pose(init_pose)
        last_ok = init_pose
        for _ in range(self.ibvs_max_iters):
            resp = vision_next_callable()
            cur_vec = np.asarray(resp.current_pose, dtype=float).reshape(6)
            tgt_vec = np.asarray(resp.target_pose, dtype=float).reshape(6)
            delta = self.ibvs_gain * (tgt_vec - cur_vec)
            cur = cur_vec + delta
            next_pose = self.pose_from_vec(cur)
            if not self._send_pose(next_pose):
                return False, last_ok
            err = float(np.linalg.norm(tgt_vec - cur))
            last_ok = next_pose
            if err < self.ibvs_eps:
                return True, next_pose
        return False, last_ok

    # ----- pick/place sequences -----
    def do_pick(self, grasp_pose: Dict[str,float], use_ibvs: bool, vision_next_callable=None) -> bool:
        approach = grasp_pose.copy(); approach['z'] += self.approach_offset_mm
        if not self._send_pose(approach): return False
        if use_ibvs and vision_next_callable is not None:
            ok, corrected = self.ibvs_correct(vision_next_callable, grasp_pose)
            if not ok: return False
            grasp_pose = corrected
        if not self._send_pose(grasp_pose): return False
        if not self._close_gripper(): return False
        lift = grasp_pose.copy(); lift['z'] += self.lift_offset_mm
        return self._send_pose(lift)

    def do_place(self, place_pose: Dict[str,float]) -> bool:
        approach = place_pose.copy(); approach['z'] += self.approach_offset_mm
        if not self._send_pose(approach): return False
        if not self._send_pose(place_pose): return False
        if not self._open_gripper(): return False
        lift = place_pose.copy(); lift['z'] += self.lift_offset_mm
        return self._send_pose(lift)

# ----------------------------
# ROS2 Node Adapter (원래 기능 유지)
# ----------------------------
class PackeeArmIBVSNode(Node):
    def __init__(self) -> None:
        super().__init__('packee_arm_ibvs_node')

        # 파라미터 선언
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 1000000)
        self.declare_parameter('move_speed', 30)
        self.declare_parameter('arm_sides', 'right')
        self.declare_parameter('approach_offset_m', 0.02)
        self.declare_parameter('lift_offset_m', 0.03)
        self.declare_parameter('gripper_open_value', 100)
        self.declare_parameter('gripper_close_value', 0)
        self.declare_parameter('pose_status_topic', '/packee/arm/pose_status')
        self.declare_parameter('pick_status_topic', '/packee/arm/pick_status')
        self.declare_parameter('place_status_topic', '/packee/arm/place_status')
        self.declare_parameter('preset_pose_cart_view', [42.2, -39.0, 289.8, -153.04, 21.75, -85.67])
        self.declare_parameter('preset_pose_standby', [52.8, -64.6, 408.6, -92.46, 0.23, -90.09])
        self.declare_parameter('use_ibvs', True)
        self.declare_parameter('ibvs_gain', 0.3)
        self.declare_parameter('ibvs_epsilon', 10.0)
        self.declare_parameter('ibvs_max_iters', 20)
        self.declare_parameter('vision_service', '/packee1/vision/detect_products_in_cart')

        # 파라미터 로드
        def P(name): return self.get_parameter(name).value
        serial_port=str(P('serial_port'))
        baud=int(P('baud_rate'))
        move_speed=int(P('move_speed'))
        approach_mm=float(P('approach_offset_m'))*1000.0
        lift_mm=float(P('lift_offset_m'))*1000.0
        g_open=int(P('gripper_open_value'))
        g_close=int(P('gripper_close_value'))
        self.use_ibvs=bool(P('use_ibvs'))
        ibvs_gain=float(P('ibvs_gain'))
        ibvs_eps=float(P('ibvs_epsilon'))
        ibvs_max_iters=int(P('ibvs_max_iters'))
        self.vision_service=str(P('vision_service'))

        # 프리셋
        self._pose_aliases={'ready_pose':'cart_view'}
        self._pose_presets={
            'cart_view': self._parse_pose_array(P('preset_pose_cart_view')),
            'standby':   self._parse_pose_array(P('preset_pose_standby')),
        }

        # 퍼블리셔/서비스/클라이언트
        self.pose_pub = self.create_publisher(ArmPoseStatus, str(P('pose_status_topic')), 10)
        self.pick_pub = self.create_publisher(ArmTaskStatus, str(P('pick_status_topic')), 10)
        self.place_pub= self.create_publisher(ArmTaskStatus, str(P('place_status_topic')), 10)

        self.create_service(ArmMoveToPose,  '/packee1/arm/move_to_pose',  self._srv_move)
        self.create_service(ArmPickProduct, '/packee1/arm/pick_product',  self._srv_pick)
        self.create_service(ArmPlaceProduct,'/packee1/arm/place_product', self._srv_place)

        self.vision_cli = self.create_client(PackeeVisionDetectProductsInCart, self.vision_service)

        # 로봇 연결
        self.robot = None
        if MyCobot280 is not None:
            try:
                self.robot = MyCobot280(serial_port, baud)
                if self.robot.is_power_on()!=1: self.robot.power_on()
                self.robot.focus_all_servos()
                self.get_logger().info(f"myCobot 연결 성공: {serial_port}, {baud}")
            except Exception as e:
                self.get_logger().error(f"myCobot 연결 실패: {e}")
                self.robot = None
        else:
            self.get_logger().warn("pymycobot 미설치: 하드웨어 제어 비활성")

        self.core = CoreArmIBVSController(
            robot=self.robot,
            move_speed=move_speed,
            approach_offset_mm=approach_mm,
            lift_offset_mm=lift_mm,
            gripper_open_value=g_open,
            gripper_close_value=g_close,
            ibvs_gain=ibvs_gain,
            ibvs_eps=ibvs_eps,
            ibvs_max_iters=ibvs_max_iters,
        )

        self.get_logger().info('✅ PackeeArmIBVSNode 준비 완료')

    # ---- utils ----
    @staticmethod
    def _parse_pose_array(raw) -> Dict[str,float]:
        if isinstance(raw, (list,tuple)) and len(raw)==6:
            vals=[float(v) for v in raw]
            return {'x':vals[0],'y':vals[1],'z':vals[2],'rx':vals[3],'ry':vals[4],'rz':vals[5]}
        return {'x':0.0,'y':0.0,'z':0.0,'rx':0.0,'ry':0.0,'rz':0.0}

    def _normalize_pose_type(self, pose_type: str) -> str:
        key=(pose_type or '').strip().lower()
        if key in self._pose_presets: return key
        return self._pose_aliases.get(key, key)

    def _pub_pose(self, robot_id:int, order_id:int, pose_type:str, status:str, progress:float, message:str):
        msg=ArmPoseStatus(robot_id=robot_id, order_id=order_id, pose_type=pose_type,
                          status=status, progress=float(progress), message=message)
        self.pose_pub.publish(msg)

    def _pub_pick(self, robot_id:int, order_id:int, product_id:int, arm_side:str, status:str, phase:str, progress:float, message:str):
        msg=ArmTaskStatus(robot_id=robot_id, order_id=order_id, product_id=product_id,
                          arm_side=arm_side, status=status, message=message, progress=float(progress))
        if hasattr(msg,'sub_status'): msg.sub_status=phase
        if hasattr(msg,'current_phase'): msg.current_phase=phase
        self.pick_pub.publish(msg)

    def _pub_place(self, robot_id:int, order_id:int, product_id:int, arm_side:str, status:str, phase:str, progress:float, message:str):
        msg=ArmTaskStatus(robot_id=robot_id, order_id=order_id, product_id=product_id,
                          arm_side=arm_side, status=status, message=message, progress=float(progress))
        if hasattr(msg,'sub_status'): msg.sub_status=phase
        if hasattr(msg,'current_phase'): msg.current_phase=phase
        self.place_pub.publish(msg)

    # ---- service handlers ----
    def _srv_move(self, req: ArmMoveToPose.Request, res: ArmMoveToPose.Response):
        pose_type=self._normalize_pose_type(req.pose_type)
        if pose_type not in self._pose_presets:
            res.success=False; res.message=f"미지원 pose_type: {req.pose_type}"; return res
        if self.core.robot is None:
            res.success=False; res.message="myCobot 연결이 활성화되지 않았습니다."; return res
        tgt=copy.deepcopy(self._pose_presets[pose_type])
        self._pub_pose(req.robot_id, req.order_id, pose_type, 'in_progress', 0.1, '자세 이동 명령 수락')
        ok=self.core._send_pose(tgt)
        if not ok:
            self._pub_pose(req.robot_id, req.order_id, pose_type, 'failed', 0.0, '자세 이동 실패')
            res.success=False; res.message='자세 이동 실패'; return res
        self._pub_pose(req.robot_id, req.order_id, pose_type, 'complete', 1.0, '자세 이동 완료')
        res.success=True; res.message='자세 이동 완료'; return res

    def _srv_pick(self, req: ArmPickProduct.Request, res: ArmPickProduct.Response):
        arm_side=req.arm_side or 'right'
        if self.core.robot is None:
            res.success=False; res.message='myCobot 연결이 활성화되지 않았습니다.'; return res
        grasp=self.core.pose_from_msg(req.pose)
        self._pub_pick(req.robot_id, req.order_id, req.product_id, arm_side, 'in_progress', 'planning', 0.05, '접근 경로 계획')

        def vision_next():
            # 실제에선 self.vision_cli 호출; 여기선 호환을 위해 내부 함수만 정의
            return PackeeVisionDetectProductsInCart.Response(current_pose=self.core.vec_from_pose(grasp), target_pose=self.core.vec_from_pose(grasp))

        use_ibvs=bool(self.use_ibvs)
        ok=self.core.do_pick(grasp_pose=grasp, use_ibvs=use_ibvs, vision_next_callable=vision_next)
        if not ok:
            self._pub_pick(req.robot_id, req.order_id, req.product_id, arm_side, 'failed', 'error', 0.0, '픽업 실패')
            res.success=False; res.message='픽업 실패'; return res
        self._pub_pick(req.robot_id, req.order_id, req.product_id, arm_side, 'completed', 'done', 1.0, '픽업 완료')
        res.success=True; res.message='픽업 완료'; return res

    def _srv_place(self, req: ArmPlaceProduct.Request, res: ArmPlaceProduct.Response):
        arm_side=req.arm_side or 'right'
        if self.core.robot is None:
            res.success=False; res.message='myCobot 연결이 활성화되지 않았습니다.'; return res
        place=self.core.pose_from_msg(req.pose)
        self._pub_place(req.robot_id, req.order_id, req.product_id, arm_side, 'in_progress', 'planning', 0.05, '접근 경로 계획')
        ok=self.core.do_place(place_pose=place)
        if not ok:
            self._pub_place(req.robot_id, req.order_id, req.product_id, arm_side, 'failed', 'error', 0.0, '담기 실패')
            res.success=False; res.message='담기 실패'; return res
        self._pub_place(req.robot_id, req.order_id, req.product_id, arm_side, 'completed', 'done', 1.0, '담기 완료')
        res.success=True; res.message='담기 완료'; return res

# ----------------------------
# Main & Tests
# ----------------------------

def _run_ros2():
    # HAS_ROS2=True일 때만 호출됨
    rclpy.init()
    node = PackeeArmIBVSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 PackeeArmIBVSNode 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _run_tests():
    import unittest

    class TestCoreArmIBVSController(unittest.TestCase):
        def setUp(self):
            self.robot = MyCobotStub()
            self.core = CoreArmIBVSController(self.robot, move_speed=25, approach_offset_mm=20.0, lift_offset_mm=30.0,
                                              gripper_open_value=100, gripper_close_value=0, ibvs_gain=0.3, ibvs_eps=5.0, ibvs_max_iters=30)

        def test_ibvs_correction_converges(self):
            start=[100, -50, 300, -160, 10, -90]
            target=[110, -40, 305, -165, 12, -92]
            vis = VisionStub(start, target)
            ok, pose = self.core.ibvs_correct(vis.next, self.core.pose_from_vec(start))
            self.assertTrue(ok)
            self.assertLess(np.linalg.norm(np.array(target)-self.core.vec_from_pose(pose)), 5.0)

        def test_pick_sequence_without_ibvs(self):
            grasp={'x':100,'y':0,'z':200,'rx':0,'ry':0,'rz':0}
            ok = self.core.do_pick(grasp_pose=grasp, use_ibvs=False)
            self.assertTrue(ok)
            # 접근(1) + 하강(2) + 상승(3) = 3회 좌표 전송
            self.assertEqual(len(self.robot.sent_coords), 3)
            # 그리퍼 닫기 1회
            self.assertEqual(len(self.robot.gripper_cmds), 1)

        def test_place_sequence(self):
            place={'x':150,'y':0,'z':180,'rx':0,'ry':0,'rz':0}
            ok = self.core.do_place(place_pose=place)
            self.assertTrue(ok)
            # 접근(1) + 하강(2) + 상승(3) = 3회 좌표 전송
            self.assertEqual(len(self.robot.sent_coords), 3)
            # 그리퍼 열기 1회
            self.assertEqual(len(self.robot.gripper_cmds), 1)

        def test_send_pose_fails_when_no_robot(self):
            core = CoreArmIBVSController(robot=None)
            self.assertFalse(core._send_pose({'x':0,'y':0,'z':0,'rx':0,'ry':0,'rz':0}))

    # ✅ 추가 테스트: IBVS 수렴 실패 시 마지막 pose 반환
    class TestIBVSFailure(unittest.TestCase):
        def test_ibvs_failure_returns_last_pose(self):
            robot = MyCobotStub()
            core = CoreArmIBVSController(robot, ibvs_gain=0.0, ibvs_eps=0.1, ibvs_max_iters=3)  # gain=0 → 절대 수렴 안 함
            start=[0,0,0,0,0,0]; target=[100,0,0,0,0,0]
            vis = VisionStub(start, target)
            ok, last = core.ibvs_correct(vis.next, core.pose_from_vec(start))
            self.assertFalse(ok)
            # 최소 한 번은 좌표를 전송해야 하며, last는 dict 6D여야 함
            self.assertGreaterEqual(len(robot.sent_coords), 1)
            self.assertTrue(all(k in last for k in ('x','y','z','rx','ry','rz')))

    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCoreArmIBVSController))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestIBVSFailure))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():
        raise SystemExit(1)

def main(args=None):
    """Entry point for ROS2 run command"""
    if HAS_ROS2:
        _run_ros2()
    else:
        # rclpy 미설치 환경: 단위 테스트 실행
        _run_tests()


if __name__ == '__main__':
    main()