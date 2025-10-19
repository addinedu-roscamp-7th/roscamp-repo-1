"""'개요' 탭의 UI 로직"""
from typing import Any, Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QLabel, QTreeWidgetItem, QTableWidgetItem, QHeaderView, QProgressBar, QWidget, QHBoxLayout

from ..ui_gen.tab_overview_ui import Ui_OverviewTab
from .base_tab import BaseTab


class OverviewTab(BaseTab, Ui_OverviewTab):
    """'개요' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        # 로봇 작업 현황 테이블 초기화
        self._setup_mission_table_columns()
        
        # 활성 주문 테이블 초기화
        self._setup_orders_table_columns()

        # 상태 트리 위젯 초기화
        self.status_tree_widget.setHeaderLabels(["항목", "값"])
        self.status_tree_widget.setColumnWidth(0, 300)
        self.robot_status_item = QTreeWidgetItem(self.status_tree_widget, ["🤖 로봇 현황"])
        self.order_status_item = QTreeWidgetItem(self.status_tree_widget, ["📦 주문 현황"])
        self.diagnostics_item = QTreeWidgetItem(self.status_tree_widget, ["🩺 시스템 진단"])
        self.robot_status_item.setExpanded(True)
        self.order_status_item.setExpanded(True)
        self.diagnostics_item.setExpanded(True)

    def _setup_mission_table_columns(self):
        """로봇 작업 현황 테이블 컬럼을 설정한다."""
        header = self.mission_queue_table.horizontalHeader()
        
        # 모든 컬럼을 균등하게 분배
        for i in range(self.mission_queue_table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

    def _setup_orders_table_columns(self):
        """활성 주문 테이블 컬럼을 설정한다."""
        header = self.active_orders_table.horizontalHeader()
        
        # 모든 컬럼을 균등하게 분배
        for i in range(self.active_orders_table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

    def _normalize_robot_type(self, robot_type) -> str:
        """
        로봇 타입을 정규화한다.

        RobotType Enum이나 문자열 모두 처리하여 소문자 문자열로 반환한다.
        """
        if robot_type is None:
            return 'unknown'

        # Enum 객체인 경우
        if hasattr(robot_type, 'value'):
            return str(robot_type.value).lower()

        # 문자열인 경우
        return str(robot_type).lower()

    def _update_robot_statistics(self, robots: list):
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

    def update_data(self, snapshot: Dict[str, Any]):
        """스냅샷 데이터로 개요 탭을 업데이트한다."""
        metrics = snapshot.get('metrics', {})
        robots = snapshot.get('robots', [])
        orders = snapshot.get('orders', {})
        
        # 로봇 통계 카드 업데이트
        self._update_robot_statistics(robots)
        
        # 로봇 작업 현황 테이블 업데이트
        self._update_mission_queue(robots, orders)
        
        # 활성 주문 테이블 업데이트
        self._update_active_orders(orders)
        
        throughput = metrics.get('hourly_throughput', 0)

        # --- 시스템 상태 요약 트리 업데이트 ---
        # 로봇 현황 업데이트
        pickee_list = [r for r in robots if self._normalize_robot_type(r.get('robot_type')) == 'pickee']
        packee_list = [r for r in robots if self._normalize_robot_type(r.get('robot_type')) == 'packee']
        
        # 업데이트 전, 확장 상태 저장
        expansion_states = {}
        for i in range(self.robot_status_item.childCount()):
            child = self.robot_status_item.child(i)
            key = child.text(0).split(' ')[0]
            expansion_states[key] = child.isExpanded()

        self.robot_status_item.takeChildren() # 기존 자식 아이템 삭제
        self._update_robot_tree_item(self.robot_status_item, "Pickee", pickee_list, robots)
        self._update_robot_tree_item(self.robot_status_item, "Packee", packee_list, robots)

        # 업데이트 후, 확장 상태 복원
        for i in range(self.robot_status_item.childCount()):
            child = self.robot_status_item.child(i)
            key = child.text(0).split(' ')[0]
            if key in expansion_states:
                child.setExpanded(expansion_states[key])

        # 주문 현황 업데이트
        order_summary = orders.get('summary', {})
        active_orders_list = orders.get('orders', [])
        self.order_status_item.takeChildren() # 기존 자식 아이템 삭제
        QTreeWidgetItem(self.order_status_item, ["진행 중 주문", f"{order_summary.get('total_active', 0)}건"])
        
        # 대기 중인 주문 수 계산
        working_robots = [r for r in robots if r.get('status') in ['WORKING', 'MOVING']]
        active_missions = len(working_robots)
        queued_orders = len(active_orders_list) - active_missions
        QTreeWidgetItem(self.order_status_item, ["대기 중 주문", f"{max(0, queued_orders)}건"])
        
        QTreeWidgetItem(self.order_status_item, ["평균 진행률", f"{order_summary.get('avg_progress', 0):.0f}%"])
        QTreeWidgetItem(self.order_status_item, ["최근 1시간 완료", f"{throughput}건"])
        QTreeWidgetItem(self.order_status_item, ["실패", f"{order_summary.get('failed_count', 0)}건"])

        # 시스템 진단 업데이트
        self._update_diagnostics_tree(metrics, robots, orders)

    def _update_robot_tree_item(self, parent_item: QTreeWidgetItem, robot_type_name: str, robot_list: list, all_robots: list):
        """로봇 현황 트리 아이템을 업데이트하는 헬퍼 함수"""
        if not robot_list:
            QTreeWidgetItem(parent_item, [f"{robot_type_name} (전체 0대)"])
            return

        active_robots = [r for r in robot_list if r.get('status') != 'OFFLINE']
        error_count = sum(1 for r in active_robots if r.get('status') == 'ERROR')

        # 배터리 통계 계산
        battery_levels = [r.get('battery_level', 0) for r in robot_list if r.get('battery_level') is not None]
        avg_battery = sum(battery_levels) / len(battery_levels) if battery_levels else 0
        low_battery_count = sum(1 for b in battery_levels if b < 30)

        type_item = QTreeWidgetItem(parent_item, [f"{robot_type_name} (활성 {len(active_robots)}/{len(robot_list)}대)"])
        QTreeWidgetItem(type_item, ["WORKING", f"{sum(1 for r in active_robots if r.get('status') == 'WORKING')}대"])
        QTreeWidgetItem(type_item, ["IDLE", f"{sum(1 for r in active_robots if r.get('status') == 'IDLE')}대"])
        error_item = QTreeWidgetItem(type_item, ["ERROR", f"{error_count}대"])
        QTreeWidgetItem(type_item, ["OFFLINE", f"{len(robot_list) - len(active_robots)}대"])
        
        # 배터리 정보 추가
        battery_item = QTreeWidgetItem(type_item, ["배터리", f"평균 {avg_battery:.1f}%, 부족 {low_battery_count}대"])
        if low_battery_count > 0:
            battery_item.setForeground(1, Qt.GlobalColor.red)

        if error_count > 0:
            error_item.setForeground(1, Qt.GlobalColor.red)
            type_item.setForeground(0, Qt.GlobalColor.red)

    def _update_diagnostics_tree(self, metrics: Dict[str, Any], robots: List[Dict[str, Any]], orders: Dict[str, Any]):
        """시스템 진단 트리 아이템을 업데이트한다."""
        self.diagnostics_item.takeChildren()

        # 성능 지표 섹션 추가
        performance_item = QTreeWidgetItem(self.diagnostics_item, ["⚡ 성능 지표"])
        
        robot_utilization = metrics.get('robot_utilization', 0)
        QTreeWidgetItem(performance_item, ["로봇 가동률", f"{robot_utilization:.1f}%"])
        
        # 로봇 활용도 계산
        working_count = sum(1 for r in robots if r.get('status') == 'WORKING')
        robot_effectiveness = (working_count / len(robots) * 100) if robots else 0
        QTreeWidgetItem(performance_item, ["로봇 활용도", f"{robot_effectiveness:.1f}%"])
        
        system_load = metrics.get('system_load', 0)
        QTreeWidgetItem(performance_item, ["시스템 부하", f"{system_load:.1f}%"])
        
        success_rate = metrics.get('success_rate', 0)
        QTreeWidgetItem(performance_item, ["성공률", f"{success_rate:.1f}%"])
        
        avg_time = metrics.get('avg_processing_time', 0)
        QTreeWidgetItem(performance_item, ["평균 완료 시간", f"{avg_time:.1f}분"])
        
        throughput = metrics.get('hourly_throughput', 0)
        QTreeWidgetItem(performance_item, ["시간당 처리량", f"{throughput}건"])
        
        performance_item.setExpanded(True)

        # 알림 섹션 추가
        alerts_item = QTreeWidgetItem(self.diagnostics_item, ["🔔 알림"])
        
        # 긴급 미션 (현재는 0으로 설정, 향후 확장 가능)
        urgent_missions = 0
        QTreeWidgetItem(alerts_item, ["긴급 미션", f"{urgent_missions}건"])
        
        # 지연된 미션 계산 (30초 이상 경과한 주문)
        active_orders_list = orders.get('orders', [])
        delayed_count = sum(1 for o in active_orders_list if o.get('elapsed_seconds', 0) > 30)
        delayed_item = QTreeWidgetItem(alerts_item, ["지연된 미션", f"{delayed_count}건"])
        if delayed_count > 0:
            delayed_item.setForeground(1, Qt.GlobalColor.red)
        
        # 배터리 경고 (30% 이하)
        battery_warning_count = sum(1 for r in robots if r.get('battery_level') is not None and r.get('battery_level') < 30)
        battery_warning_item = QTreeWidgetItem(alerts_item, ["배터리 경고", f"{battery_warning_count}대"])
        if battery_warning_count > 0:
            battery_warning_item.setForeground(1, Qt.GlobalColor.red)
        
        alerts_item.setExpanded(True)

        # 실패/오류 섹션
        failures_item = QTreeWidgetItem(self.diagnostics_item, ["🩺 실패/오류"])
        failed_by_reason = metrics.get('failed_orders_by_reason', {})
        if failed_by_reason:
            reason_text = ', '.join(f'{reason}: {count}건' for reason, count in failed_by_reason.items())
        else:
            reason_text = '없음'
        QTreeWidgetItem(failures_item, ["최근 실패 주문(60분)", reason_text])

        failed_orders = metrics.get('failed_orders', [])
        if failed_orders:
            for order in failed_orders:
                ended_at = order.get('ended_at')
                ended_text = ended_at[11:19] if isinstance(ended_at, str) and 'T' in ended_at else '-'
                amount = order.get('total_price')
                amount_text = f'₩{int(amount):,}' if amount else '-'
                order_line = f"#{order.get('order_id', '-')} / 사유={order.get('failure_reason', 'UNKNOWN')} / 금액={amount_text} / 종료={ended_text}"
                QTreeWidgetItem(failures_item, [order_line])

        # LLM 상태를 트리 구조로 변경
        llm_stats = metrics.get('llm_stats', {})
        llm_item = QTreeWidgetItem(failures_item, ["LLM 상태"])
        QTreeWidgetItem(llm_item, ["성공률", f"{llm_stats.get('success_rate', 0.0):.1f}%"])
        QTreeWidgetItem(llm_item, ["평균 응답 시간", f"{llm_stats.get('avg_response_time', 0.0):.1f}ms"])
        QTreeWidgetItem(llm_item, ["폴백 횟수", f"{llm_stats.get('fallback_count', 0)}회"])
        QTreeWidgetItem(llm_item, ["실패 횟수", f"{llm_stats.get('failure_count', 0)}회"])
        llm_item.setExpanded(True)

        QTreeWidgetItem(failures_item, ["ROS 서비스 재시도", f"{metrics.get('ros_retry_count', 0)}회"])

        # 로봇 장애 섹션
        error_robots_item = QTreeWidgetItem(failures_item, ["로봇 장애"])
        error_robots = metrics.get('error_robots', [])
        if not error_robots:
            QTreeWidgetItem(error_robots_item, ["없음"])
        else:
            for robot in error_robots:
                line = self._format_robot_line(robot, '오류')
                QTreeWidgetItem(error_robots_item, [line])

        # 네트워크/연결 섹션
        network_item = QTreeWidgetItem(failures_item, ["네트워크/연결"])
        network = metrics.get('network', {})
        app_sessions = network.get('app_sessions', 0)
        app_max = network.get('app_sessions_max', 200)
        QTreeWidgetItem(network_item, ["App 세션", f"{app_sessions} / {app_max}"])
        
        llm_response = network.get('llm_response_time', 0)
        QTreeWidgetItem(network_item, ["LLM 응답 시간", f"{llm_response:.0f}ms"])

        failures_item.setExpanded(True)
        error_robots_item.setExpanded(True)
        network_item.setExpanded(True)

    @staticmethod
    def _format_robot_line(robot: Dict[str, Any], label: str) -> str:
        """로봇 장애 정보를 문자열로 변환한다."""
        robot_id = robot.get('robot_id', '-')
        robot_type = robot.get('robot_type', '-')
        status = robot.get('status', '-')
        last_update = robot.get('last_update')
        if isinstance(last_update, str) and 'T' in last_update:
            last_seen = last_update[11:19]
        else:
            last_seen = '-'
        return f"#{robot_id} ({robot_type}) [{label}] 상태={status} / 마지막 갱신={last_seen}"

    def _update_mission_queue(self, robots: List[Dict[str, Any]], orders: Dict[str, Any]):
        """로봇 작업 현황 테이블을 업데이트한다."""
        tasks = []
        
        # 현재 작업 중인 로봇들의 작업
        for robot in robots:
            robot_id = robot.get('robot_id')
            robot_type = robot.get('robot_type', 'UNKNOWN')
            status = robot.get('status', 'UNKNOWN')
            active_order_id = robot.get('active_order_id')
            battery = robot.get('battery_level', 0)
            
            if status in ['WORKING', 'MOVING']:
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
                    'order_id': active_order_id if active_order_id else '-',
                    'status': '진행 중' if active_order_id else status,
                    'battery': battery,
                })
            elif status == 'IDLE':
                tasks.append({
                    'robot_id': robot_id,
                    'task_type': '대기 중',
                    'order_id': '-',
                    'status': '대기',
                    'battery': battery,
                })
            elif status == 'CHARGING':
                tasks.append({
                    'robot_id': robot_id,
                    'task_type': '충전 중',
                    'order_id': '-',
                    'status': '충전 중',
                    'battery': battery,
                })
            elif status == 'ERROR':
                tasks.append({
                    'robot_id': robot_id,
                    'task_type': '오류',
                    'order_id': '-',
                    'status': 'ERROR',
                    'battery': battery,
                })
            elif status == 'OFFLINE':
                tasks.append({
                    'robot_id': robot_id,
                    'task_type': '오프라인',
                    'order_id': '-',
                    'status': 'OFFLINE',
                    'battery': battery,
                })
            else:
                # 정의되지 않은 상태도 표시 (누락 방지)
                tasks.append({
                    'robot_id': robot_id,
                    'task_type': str(status),
                    'order_id': active_order_id if active_order_id else '-',
                    'status': str(status),
                    'battery': battery,
                })
        
        # 로봇 ID 순으로 정렬 (작업 중인 로봇 우선)
        tasks.sort(key=lambda x: (
            0 if x['status'] == '진행 중' else 1,
            x['robot_id'] if isinstance(x['robot_id'], int) else 999
        ))
        
        # 최대 15개까지만 표시
        tasks = tasks[:15]
        
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
            
            # 컬럼 4: 배터리
            battery = task.get('battery', 0)
            battery_item = QTableWidgetItem(f"{battery:.1f}%" if battery is not None else '-')
            if battery is not None:
                if battery < 20:
                    battery_item.setForeground(Qt.GlobalColor.red)
                elif battery < 50:
                    battery_item.setForeground(Qt.GlobalColor.darkYellow)
                else:
                    battery_item.setForeground(Qt.GlobalColor.green)
            self.mission_queue_table.setItem(row, 4, battery_item)

    def _update_active_orders(self, orders: Dict[str, Any]):
        """활성 주문 테이블을 업데이트한다."""
        # 'orders' 키에서 주문 리스트를 가져옴 (주문 관리 탭과 동일)
        active_orders = orders.get('orders', [])
        
        # 최대 10개까지만 표시
        active_orders = active_orders[:10]
        
        self.active_orders_table.setRowCount(len(active_orders))
        
        for row, order in enumerate(active_orders):
            # 컬럼 0: Order ID
            self.active_orders_table.setItem(row, 0, QTableWidgetItem(str(order.get('order_id', ''))))
            
            # 컬럼 1: Status
            status = str(order.get('status', ''))
            status_item = QTableWidgetItem(status)
            if 'FAIL' in status:
                status_item.setForeground(Qt.GlobalColor.red)
            elif status in ['PACKED', 'DELIVERED']:
                status_item.setForeground(Qt.GlobalColor.green)
            elif status in ['PICKING', 'PACKING', 'MOVING']:
                status_item.setForeground(Qt.GlobalColor.blue)
            self.active_orders_table.setItem(row, 1, status_item)
            
            # 컬럼 2: Items
            self.active_orders_table.setItem(row, 2, QTableWidgetItem(str(order.get('total_items', 0))))
            
            # 컬럼 3: Amount
            total_price = order.get('total_price')
            amount_text = f'₩{int(total_price):,}' if total_price is not None else '-'
            self.active_orders_table.setItem(row, 3, QTableWidgetItem(amount_text))
            
            # 컬럼 4: Progress (프로그레스 바로 표시)
            progress = order.get('progress', 0)
            progress_widget = self._create_progress_widget(progress)
            self.active_orders_table.setCellWidget(row, 4, progress_widget)
            
            # 컬럼 5: Elapsed
            elapsed_sec = order.get('elapsed_seconds')
            elapsed_text = f'{int(elapsed_sec // 60)}m {int(elapsed_sec % 60)}s' if elapsed_sec is not None else '-'
            elapsed_item = QTableWidgetItem(elapsed_text)
            if elapsed_sec and elapsed_sec > 30:
                elapsed_item.setForeground(Qt.GlobalColor.red)
            elif elapsed_sec and elapsed_sec > 20:
                elapsed_item.setForeground(Qt.GlobalColor.darkYellow)
            self.active_orders_table.setItem(row, 5, elapsed_item)

    def _create_progress_widget(self, progress: float) -> QWidget:
        """프로그레스 바 위젯을 생성한다."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        
        progress_bar = QProgressBar()
        progress_bar.setValue(int(progress))
        progress_bar.setFormat(f'{progress:.0f}%')
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 2px;
            }
        """)
        
        layout.addWidget(progress_bar)
        return container

    def add_alert(self, event_data: Dict[str, Any]):
        """
        최근 알림 처리 (개요 탭에서는 더 이상 사용하지 않음)
        
        이벤트 로그 탭으로 이동됨. 하위 호환성을 위해 빈 함수로 유지.
        """
        pass

    def add_ros_topic_event(self, event_data: Dict[str, Any]):
        """
        ROS 토픽 이벤트 처리 (개요 탭에서는 더 이상 사용하지 않음)
        
        이벤트 로그 탭으로 이동됨. 하위 호환성을 위해 빈 함수로 유지.
        """
        pass

