"""'ROS2 토픽 모니터' 탭의 UI 로직"""
import json
from datetime import datetime
from typing import Any, Dict

from PyQt6.QtGui import QTextCursor

from ..ui_gen.tab_topic_monitor_ui import Ui_TopicMonitorTab
from .base_tab import BaseTab


class TopicMonitorTab(BaseTab, Ui_TopicMonitorTab):
    """'ROS2 토픽 모니터' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        from ...constants import MAX_TOPIC_LOG_ENTRIES

        super().__init__(parent)
        self.setupUi(self)
        self.MAX_LOG_ENTRIES = MAX_TOPIC_LOG_ENTRIES

    def add_ros_topic_event(self, event_data: Dict[str, Any]):
        """ROS 토픽 수신 이벤트를 해당 영역에 표시한다."""
        topic_name = event_data.get('topic_name', 'unknown')
        is_periodic = event_data.get('is_periodic', False)
        timestamp = event_data.get('timestamp', 0)
        msg_dict = event_data.get('msg', {})

        dt_object = datetime.fromtimestamp(timestamp)
        time_str = dt_object.strftime('%H:%M:%S.%f')[:-3]

        # 메시지 내용을 예쁘게 포맷
        msg_str = json.dumps(msg_dict, indent=2, ensure_ascii=False)

        log_line = f"[{time_str}] {topic_name}\n{msg_str}\n"

        if is_periodic:
            target_widget = self.periodic_log
        else:
            target_widget = self.event_log

        target_widget.append(log_line)

        # 최대 라인 수 유지
        if target_widget.document().lineCount() > self.MAX_LOG_ENTRIES * 2: # (내용 + 빈 줄)
            cursor = target_widget.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor, 2)
            cursor.removeSelectedText()
            cursor.deletePreviousChar() # 줄바꿈 제거

    def update_data(self, data):
        """이 탭은 스냅샷 데이터를 사용하지 않고 이벤트만 받습니다."""
        pass
