"""'이벤트 로그' 탭의 UI 로직"""
from typing import Any, Dict
from datetime import datetime

from PyQt6.QtGui import QTextCursor

from ..ui_gen.tab_event_log_ui import Ui_EventLogTab
from .base_tab import BaseTab


class EventLogTab(BaseTab, Ui_EventLogTab):
    """'이벤트 로그' 탭의 UI 및 로직"""

    MAX_LOG_ENTRIES = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

    def add_event(self, event_data: Dict[str, Any]):
        """이벤트 로그 항목 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_line = f"[{timestamp}] {event_data.get('type', 'unknown')}: {event_data.get('message', '')}"
        self.log_text_edit.append(log_line)
        if self.log_text_edit.document().lineCount() > self.MAX_LOG_ENTRIES:
            cursor = self.log_text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()

    def update_data(self, data):
        """이 탭은 스냅샷 데이터를 사용하지 않고 이벤트만 받습니다."""
        pass
