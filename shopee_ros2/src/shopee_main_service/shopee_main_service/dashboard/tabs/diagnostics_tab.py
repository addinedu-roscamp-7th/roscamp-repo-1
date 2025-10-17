'''시스템 진단' 탭의 UI 로직'''
from typing import Any, Dict

from ..ui_gen.tab_diagnostics_ui import Ui_SystemDiagnosticsTab
from .base_tab import BaseTab


class SystemDiagnosticsTab(BaseTab, Ui_SystemDiagnosticsTab):
    """'시스템 진단' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        # 프로그레스 바 초기 설정 (테스트용)
        self.app_sessions_progress.setVisible(True)
        self.llm_response_progress.setVisible(True) 
        self.db_connections_progress.setVisible(True)
        
        # 테스트용 기본값 설정
        self.app_sessions_progress.setValue(25)
        self.llm_response_progress.setValue(50)
        self.db_connections_progress.setValue(30)

    def update_data(self, snapshot: Dict[str, Any]):
        """스냅샷 데이터로 진단 탭을 업데이트한다."""
        metrics = snapshot.get('metrics', {})
        failed_by_reason = metrics.get('failed_orders_by_reason', {})
        failed_orders = metrics.get('failed_orders', [])
        error_robots = metrics.get('error_robots', [])
        offline_robots = metrics.get('offline_robots', [])
        llm_stats = metrics.get('llm_stats', {})
        ros_retry_count = metrics.get('ros_retry_count', 0)

        # 실패 주문 통계
        if failed_by_reason:
            reason_text = ', '.join(f'{reason}: {count}건' for reason, count in failed_by_reason.items())
        else:
            reason_text = '없음'

        # 최근 실패 주문 상세
        if failed_orders:
            order_lines = []
            for order in failed_orders:
                ended_at = order.get('ended_at')
                ended_text = ended_at[11:19] if isinstance(ended_at, str) and 'T' in ended_at else '-'
                amount = order.get('total_price')
                amount_text = f'₩{int(amount):,}' if amount else '-'
                order_lines.append(
                    f"  #{order.get('order_id', '-')} / 사유={order.get('failure_reason', 'UNKNOWN')} / 금액={amount_text} / 종료={ended_text}"
                )
            failed_orders_text = '\n'.join(order_lines)
        else:
            failed_orders_text = '  없음'

        # 로봇 장애 목록
        robot_lines = []
        for robot in error_robots:
            robot_lines.append(self._format_robot_line(robot, '오류'))
        for robot in offline_robots:
            robot_lines.append(self._format_robot_line(robot, '오프라인'))
        if not robot_lines:
            robot_lines.append('  없음')

        llm_success = llm_stats.get('success_rate', 0.0)
        llm_latency = llm_stats.get('avg_response_time', 0.0)
        fallback_count = llm_stats.get('fallback_count', 0)
        failure_count = llm_stats.get('failure_count', 0)

        error_lines = [
            f'최근 실패 주문(60분): {reason_text}',
            failed_orders_text,
            '',
            '로봇 장애 현황:',
            *robot_lines,
            '',
            f'LLM 상태: 성공률 {llm_success:.1f}% / 응답 {llm_latency:.1f}ms / 폴백 {fallback_count}회 / 실패 {failure_count}회',
            f'ROS 서비스 재시도: {ros_retry_count}회',
        ]
        self.error_label.setText('\n'.join(error_lines))

        # 네트워크 및 연결 상태
        network = metrics.get('network', {})
        app_sessions = network.get('app_sessions', 0)
        app_max = network.get('app_sessions_max', 0)
        topic_health = network.get('ros_topic_health', {})
        receive_rate = network.get('topic_receive_rate', 0.0)
        event_activity = network.get('event_topic_activity', {})
        event_timeout = network.get('event_topic_timeout')
        db_connections = network.get('db_connections', 0)
        db_max = network.get('db_connections_max', 0)
        llm_response = network.get('llm_response_time', 0.0)

        network_lines = [
            f'App 세션: {app_sessions} / {app_max}',
        ]

        if topic_health:
            unhealthy_topics = [name for name, healthy in topic_health.items() if not healthy]
            if unhealthy_topics:
                network_lines.append(f"ROS 토픽: 오류 ({', '.join(unhealthy_topics)})")
            else:
                network_lines.append(f'ROS 토픽: 정상 ({receive_rate:.1f}% 수신)')
        else:
            network_lines.append('ROS 토픽: 정보 없음')

        if event_activity:
            overdue_topics = [name for name, info in event_activity.items() if info.get('overdue')]
            if overdue_topics:
                network_lines.append(f"이벤트 토픽 지연: {', '.join(overdue_topics)}")
            if event_timeout:
                network_lines.append(f'이벤트 토픽 지연 임계값: {event_timeout:.1f}s')
            activity_lines = []
            for name, info in event_activity.items():
                seconds_since_last = info.get('seconds_since_last')
                if seconds_since_last is None:
                    activity_lines.append(f'{name}=미수신')
                else:
                    activity_lines.append(f'{name}={seconds_since_last:.1f}초')
            if activity_lines:
                network_lines.append('이벤트 토픽 최신 수신: ' + ', '.join(activity_lines))

        network_lines.append(f'LLM 응답 시간: {llm_response:.1f}ms')
        network_lines.append(f'DB 커넥션: {db_connections} / {db_max}')

        # 프로그레스 바 업데이트
        self._update_progress_bars(network)
        
        # 상세 정보 업데이트
        self.network_details_label.setText('\n'.join(network_lines))

    def _update_progress_bars(self, network: Dict[str, Any]) -> None:
        """네트워크 상태 프로그레스 바를 업데이트한다."""
        # App 세션 프로그레스 바
        app_sessions = network.get('app_sessions', 0)
        app_max = network.get('app_sessions_max', 200)
        app_percentage = (app_sessions / app_max * 100) if app_max > 0 else 0
        
        self.app_sessions_label.setText(f'App 세션: {app_sessions} / {app_max}')
        self.app_sessions_progress.setMaximum(100)
        self.app_sessions_progress.setValue(int(app_percentage))
        
        # 임계값에 따른 색상 설정
        if app_percentage >= 90:
            self.app_sessions_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #f44336; 
                    border-radius: 2px;
                }  /* 빨강 */
            """)
        elif app_percentage >= 70:
            self.app_sessions_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #ff9800; 
                    border-radius: 2px;
                }  /* 주황 */
            """)
        else:
            self.app_sessions_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #4caf50; 
                    border-radius: 2px;
                }  /* 녹색 */
            """)
        
        # LLM 응답 시간 프로그레스 바
        llm_response = network.get('llm_response_time', 0)
        llm_max = 1500  # 1500ms 임계값
        llm_percentage = min((llm_response / llm_max * 100), 100) if llm_max > 0 else 0
        
        self.llm_response_label.setText(f'LLM 응답 시간: {llm_response:.0f}ms / {llm_max}ms')
        self.llm_response_progress.setMaximum(100)
        self.llm_response_progress.setValue(int(llm_percentage))
        
        # LLM 응답 시간 색상 설정
        if llm_response >= 1500:
            self.llm_response_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #f44336; 
                    border-radius: 2px;
                }  /* 빨강 */
            """)
        elif llm_response >= 1000:
            self.llm_response_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #ff9800; 
                    border-radius: 2px;
                }  /* 주황 */
            """)
        else:
            self.llm_response_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #4caf50; 
                    border-radius: 2px;
                }  /* 녹색 */
            """)
        
        # DB 커넥션 프로그레스 바
        db_connections = network.get('db_connections', 0)
        db_max = network.get('db_connections_max', 10)
        db_percentage = (db_connections / db_max * 100) if db_max > 0 else 0
        
        self.db_connections_label.setText(f'DB 커넥션: {db_connections} / {db_max}')
        self.db_connections_progress.setMaximum(100)
        self.db_connections_progress.setValue(int(db_percentage))
        
        # DB 커넥션 색상 설정
        if db_percentage >= 90:
            self.db_connections_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #f44336; 
                    border-radius: 2px;
                }  /* 빨강 */
            """)
        elif db_percentage >= 70:
            self.db_connections_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #ff9800; 
                    border-radius: 2px;
                }  /* 주황 */
            """)
        else:
            self.db_connections_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk { 
                    background-color: #4caf50; 
                    border-radius: 2px;
                }  /* 녹색 */
            """)

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
        return f"  #{robot_id} ({robot_type}) [{label}] 상태={status} / 마지막 갱신={last_seen}"
