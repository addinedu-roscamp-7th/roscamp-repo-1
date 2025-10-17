"""'로봇 관제' 탭의 UI 로직"""
from typing import Any, Dict, List
from datetime import datetime, timedelta

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem, QHeaderView, QProgressBar, QWidget, QHBoxLayout

from ..ui_gen.tab_robot_control_ui import Ui_RobotControlTab
from .base_tab import BaseTab


class RobotControlTab(BaseTab, Ui_RobotControlTab):
    """'로봇 관제' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._setup_table_columns()
        self._mission_queue_data: List[Dict[str, Any]] = []

    def _normalize_robot_type(self, robot_type) -> str:
        """
        로봇 타입을 정규화한다.

        RobotType Enum이나 문자열 모두 처리하여 소문자 문자열로 반환한다.
        - RobotType.PICKEE → 'pickee'
        - 'PICKEE' → 'pickee'
        - 'pickee' → 'pickee'
        """
        if robot_type is None:
            return 'unknown'

        # Enum 객체인 경우
        if hasattr(robot_type, 'value'):
            return str(robot_type.value).lower()

        # 문자열인 경우
        return str(robot_type).lower()

    def _setup_table_columns(self):
        """테이블 컬럼 너비와 리사이즈 정책을 설정한다."""
        header = self.mission_queue_table.horizontalHeader()

        # 각 컬럼의 너비를 설정 (픽셀 단위)
        self.mission_queue_table.setColumnWidth(0, 80)   # Robot ID
        self.mission_queue_table.setColumnWidth(1, 120)  # 작업 타입
        self.mission_queue_table.setColumnWidth(2, 90)   # Order ID
        self.mission_queue_table.setColumnWidth(3, 80)   # 상태
        self.mission_queue_table.setColumnWidth(4, 120)  # 위치

        # 마지막 컬럼(배터리)은 남은 공간을 모두 차지하도록 설정
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        # 나머지 컬럼들은 고정 크기로 설정
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)

    def update_data(self, snapshot: Dict[str, Any]):
        """스냅샷 데이터로 로봇 관제 탭을 업데이트한다."""
        robots = snapshot.get('robots', [])
        orders = snapshot.get('orders', {})
        metrics = snapshot.get('metrics', {})
        
        # 로봇 통계 업데이트
        self._update_robot_statistics(robots)
        
        # 미션 큐 업데이트
        self._update_mission_queue(robots, orders)
        
        # 성능 지표 업데이트
        self._update_performance_metrics(robots, orders, metrics)

    def _update_robot_statistics(self, robots: List[Dict[str, Any]]):
        """로봇 통계 카드들을 업데이트한다."""
        total_robots = len(robots)
        pickee_count = sum(1 for r in robots if self._normalize_robot_type(r.get('robot_type')) == 'pickee')
        packee_count = sum(1 for r in robots if self._normalize_robot_type(r.get('robot_type')) == 'packee')
        
        # 상태별 로봇 수 계산
        available_robots = sum(1 for r in robots if r.get('status') == 'IDLE' and not r.get('reserved', False))
        working_robots = sum(1 for r in robots if r.get('status') in ['WORKING', 'MOVING'])
        charging_robots = sum(1 for r in robots if r.get('status') == 'CHARGING')
        offline_robots = sum(1 for r in robots if r.get('status') in ['OFFLINE', 'ERROR'])
        
        # 전체 로봇 카드 업데이트
        self.total_robots_count.setText(f"{total_robots}대")
        self.total_robots_detail.setText(f"Pickee: {pickee_count} | Packee: {packee_count}")
        
        # 가용 로봇 카드 업데이트
        self.available_robots_count.setText(f"{available_robots}대")
        availability_rate = (available_robots / total_robots * 100) if total_robots > 0 else 0
        self.available_robots_detail.setText(f"가용률: {availability_rate:.1f}%")
        
        # 작업 중 로봇 카드 업데이트
        self.working_robots_count.setText(f"{working_robots}대")
        utilization_rate = (working_robots / total_robots * 100) if total_robots > 0 else 0
        self.working_robots_detail.setText(f"가동률: {utilization_rate:.1f}% | 충전: {charging_robots}대")
        
        # 오프라인 로봇 카드 업데이트
        self.offline_robots_count.setText(f"{offline_robots}대")
        offline_rate = (offline_robots / total_robots * 100) if total_robots > 0 else 0
        self.offline_robots_detail.setText(f"오프라인율: {offline_rate:.1f}%")

    def _update_mission_queue(self, robots: List[Dict[str, Any]], orders: Dict[str, Any]):
        """작업 현황 테이블을 업데이트한다."""
        tasks = self._generate_mission_queue_data(robots, orders)
        
        self.mission_queue_table.setRowCount(len(tasks))
        
        for row, task in enumerate(tasks):
            # 컬럼 0: Robot ID
            robot_id = task.get('robot_id', '-')
            robot_item = QTableWidgetItem(str(robot_id))
            if task.get('status') == '진행 중':
                robot_item.setForeground(Qt.GlobalColor.blue)
            elif task.get('status') in ['ERROR', 'OFFLINE']:
                robot_item.setForeground(Qt.GlobalColor.red)
            self.mission_queue_table.setItem(row, 0, robot_item)
            
            # 컬럼 1: Task Type
            task_type = task.get('task_type', 'UNKNOWN')
            self.mission_queue_table.setItem(row, 1, QTableWidgetItem(task_type))
            
            # 컬럼 2: Order ID
            order_id = task.get('order_id', '-')
            self.mission_queue_table.setItem(row, 2, QTableWidgetItem(str(order_id)))
            
            # 컬럼 3: Status
            status = task.get('status', 'UNKNOWN')
            status_item = QTableWidgetItem(status)
            if status == '진행 중':
                status_item.setForeground(Qt.GlobalColor.green)
            elif status in ['ERROR', 'OFFLINE']:
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == '대기':
                status_item.setForeground(Qt.GlobalColor.blue)
            self.mission_queue_table.setItem(row, 3, status_item)
            
            # 컬럼 4: Location
            location = task.get('location', 'UNKNOWN')
            self.mission_queue_table.setItem(row, 4, QTableWidgetItem(location))

            # 컬럼 5: Battery Level
            battery = self._get_robot_battery(robots, task.get('robot_id'))
            battery_item = QTableWidgetItem(f"{battery:.1f}%" if battery is not None else '-')
            if battery is not None:
                if battery < 20:
                    battery_item.setForeground(Qt.GlobalColor.red)
                elif battery < 50:
                    battery_item.setForeground(Qt.GlobalColor.darkYellow)
                else:
                    battery_item.setForeground(Qt.GlobalColor.green)
            self.mission_queue_table.setItem(row, 5, battery_item)

    def _generate_mission_queue_data(self, robots: List[Dict[str, Any]], orders: Dict[str, Any]) -> List[Dict[str, Any]]:
        """로봇과 주문 데이터를 기반으로 작업 현황 데이터를 생성한다."""
        tasks = []
        
        # 현재 작업 중인 로봇들의 작업
        for robot in robots:
            robot_id = robot.get('robot_id')
            robot_type = robot.get('robot_type', 'UNKNOWN')
            status = robot.get('status', 'UNKNOWN')
            active_order_id = robot.get('active_order_id')
            
            if status in ['WORKING', 'MOVING'] and active_order_id:
                # 작업 타입 결정
                normalized_type = self._normalize_robot_type(robot_type)
                if normalized_type == 'pickee':
                    task_type = '상품 픽업'
                elif normalized_type == 'packee':
                    task_type = '상품 포장'
                else:
                    task_type = '작업 중'

                tasks.append({
                    'robot_id': robot_id,
                    'task_type': task_type,
                    'order_id': active_order_id,
                    'location': self._get_current_location(robot),
                    'status': '진행 중',
                })
            elif status == 'IDLE':
                tasks.append({
                    'robot_id': robot_id,
                    'task_type': '대기 중',
                    'order_id': '-',
                    'location': self._get_current_location(robot),
                    'status': '대기',
                })
            elif status == 'CHARGING':
                tasks.append({
                    'robot_id': robot_id,
                    'task_type': '충전 중',
                    'order_id': '-',
                    'location': '충전 구역',
                    'status': '충전 중',
                })
            elif status in ['ERROR', 'OFFLINE']:
                tasks.append({
                    'robot_id': robot_id,
                    'task_type': '오류/오프라인',
                    'order_id': '-',
                    'location': '알 수 없음',
                    'status': status,
                })
        
        # 대기 중인 주문들 (로봇에 할당되지 않은 주문)
        active_orders = orders.get('active_orders', [])
        assigned_order_ids = {t.get('order_id') for t in tasks if t.get('order_id') != '-'}
        
        waiting_orders = [order for order in active_orders if order.get('order_id') not in assigned_order_ids]
        
        for order in waiting_orders[:5]:  # 최대 5개만 표시
            order_id = order.get('order_id')
            tasks.append({
                'robot_id': '할당 대기',
                'task_type': '픽업 대기',
                'order_id': order_id,
                'location': '대기열',
                'status': '대기',
            })
        
        # 로봇 ID 순으로 정렬 (작업 중인 로봇 우선)
        tasks.sort(key=lambda x: (
            0 if x['status'] == '진행 중' else 1,  # 진행 중인 작업 우선
            x['robot_id'] if isinstance(x['robot_id'], int) else 999
        ))
        
        return tasks

    def _get_current_location(self, robot: Dict[str, Any]) -> str:
        """로봇의 현재 위치를 반환한다."""
        # current_location 필드가 있으면 사용
        location = robot.get('current_location')
        if location:
            return str(location)

        # 없으면 상태 기반으로 기본값
        status = robot.get('status', 'UNKNOWN')
        if status == 'CHARGING':
            return '충전 구역'
        elif status in ['ERROR', 'OFFLINE']:
            return '알 수 없음'

        return '-'

    def _get_robot_battery(self, robots: List[Dict[str, Any]], robot_id) -> float:
        """특정 로봇의 배터리 레벨을 반환한다."""
        if isinstance(robot_id, str) or robot_id == '-':
            return None
            
        for robot in robots:
            if robot.get('robot_id') == robot_id:
                return robot.get('battery_level')
        return None

    def _update_performance_metrics(self, robots: List[Dict[str, Any]], orders: Dict[str, Any], metrics: Dict[str, Any]):
        """성능 지표를 업데이트한다."""
        total_robots = len(robots)
        working_robots = sum(1 for r in robots if r.get('status') in ['WORKING', 'MOVING'])
        available_robots = sum(1 for r in robots if r.get('status') == 'IDLE')
        charging_robots = sum(1 for r in robots if r.get('status') == 'CHARGING')
        
        # 미션 통계
        active_missions = len([r for r in robots if r.get('status') == 'WORKING'])
        queued_missions = len(orders.get('active_orders', [])) - active_missions
        
        # 배터리 통계
        battery_levels = [r.get('battery_level', 0) for r in robots if r.get('battery_level') is not None]
        avg_battery = sum(battery_levels) / len(battery_levels) if battery_levels else 0
        low_battery_count = sum(1 for b in battery_levels if b < 30)
        
        # 퍼센트 계산 (division by zero 방지)
        working_pct = (working_robots / total_robots * 100) if total_robots > 0 else 0
        available_pct = (available_robots / total_robots * 100) if total_robots > 0 else 0
        charging_pct = (charging_robots / total_robots * 100) if total_robots > 0 else 0
        
        # 성능 지표 텍스트 생성
        now = datetime.now()
        performance_text = f"""=== 로봇 관제 성능 지표 ===
업데이트 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}

🤖 로봇 현황:
• 전체 로봇: {total_robots}대
• 작업 중: {working_robots}대 ({working_pct:.1f}%)
• 가용 상태: {available_robots}대 ({available_pct:.1f}%)
• 충전 중: {charging_robots}대 ({charging_pct:.1f}%)
• 평균 배터리: {avg_battery:.1f}%
• 배터리 부족 로봇: {low_battery_count}대

📋 미션 현황:
• 진행 중인 미션: {active_missions}건
• 대기 중인 미션: {queued_missions}건
• 평균 미션 완료 시간: {metrics.get('avg_processing_time', 0):.1f}분
• 시간당 처리량: {metrics.get('hourly_throughput', 0)}건

⚡ 실시간 지표:
• 로봇 가동률: {metrics.get('robot_utilization', 0):.1f}%
• 시스템 부하: {metrics.get('system_load', 0):.1f}%
• 성공률: {metrics.get('success_rate', 0):.1f}%

🔔 알림:
• 긴급 미션: {sum(1 for m in self._mission_queue_data if m.get('priority') == 'URGENT')}건
• 지연된 미션: {sum(1 for m in self._mission_queue_data if m.get('status') == 'DELAYED')}건
• 배터리 경고: {low_battery_count}건

📊 효율성 분석:
• 로봇 활용도: {self._calculate_robot_utilization(robots):.1f}%
        """.strip()
        
        self.performance_metrics.setPlainText(performance_text)

    def _calculate_robot_utilization(self, robots: List[Dict[str, Any]]) -> float:
        """로봇 활용도를 계산한다."""
        if not robots:
            return 0.0

        working_count = sum(1 for r in robots if r.get('status') == 'WORKING')
        return (working_count / len(robots)) * 100
