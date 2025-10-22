import json
from typing import Any, Dict

from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from PyQt6.QtWidgets import QWidget

from ...constants import EventTopic
from ..ui_gen.tab_tcp_monitor_ui import Ui_TCPMonitorTab


class TCPMonitorTab(QWidget, Ui_TCPMonitorTab):
    """TCP 통신 내용을 표시하는 탭 (UI 파일 사용)"""

    message_signal = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        # 시그널-슬롯 연결
        self.message_signal.connect(self._append_message)

    def handle_tcp_event(self, event_data: Dict[str, Any]) -> None:
        """메인 윈도우로부터 TCP 이벤트를 받아 처리"""
        event_type = event_data.get("topic")
        message = event_data.get("data", {})
        
        if event_type == EventTopic.TCP_MESSAGE_RECEIVED.value:
            self._format_and_emit("received", message)
        elif event_type == EventTopic.TCP_MESSAGE_SENT.value:
            self._format_and_emit("sent", message)

    def _format_and_emit(self, tab_name: str, message: Dict[str, Any]) -> None:
        """메시지 포맷팅 및 시그널 발생"""
        timestamp = message.get("timestamp", "")
        peer = message.get("peer", "")
        payload_str = message.get("payload", "")

        try:
            payload_json = json.loads(payload_str)
            formatted_payload = json.dumps(payload_json, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            formatted_payload = payload_str

        full_text = f"[{timestamp}] From/To: {peer}\n{formatted_payload}\n{'-'*40}\n"
        self.message_signal.emit(tab_name, full_text)

    @Slot(str, str)
    def _append_message(self, tab_name: str, text: str) -> None:
        """시그널을 받아 스레드에 안전하게 UI 업데이트"""
        if tab_name == "received":
            # self.received_text는 .ui 파일에 정의된 objectName
            self.received_text.appendPlainText(text)
        elif tab_name == "sent":
            # self.sent_text는 .ui 파일에 정의된 objectName
            self.sent_text.appendPlainText(text)
