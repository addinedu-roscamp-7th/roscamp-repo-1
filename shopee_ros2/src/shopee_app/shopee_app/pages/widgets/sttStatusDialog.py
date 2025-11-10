from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from PyQt6 import QtCore
from PyQt6.QtWidgets import QLabel


class SttStatusDialog(QDialog):
    """음성 인식 진행 상황을 실시간으로 보여 주는 모달 대화상자."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("음성 인식")
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowType.Tool)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        self.message_label = QLabel("", self)
        layout.addWidget(self.message_label)

    def update_message(self, text: str, *, warning: bool = False) -> None:
        """상태 메시지를 갱신하고 경고 여부에 따라 색상을 조정한다."""
        # 안내 문구를 업데이트하면서 경고 여부에 따라 색상을 바꾼다.
        self.message_label.setText(text)
        if warning:
            self.message_label.setStyleSheet("color: #d32f2f;")
        else:
            self.message_label.setStyleSheet("")
        self.adjustSize()
        self.repaint()
