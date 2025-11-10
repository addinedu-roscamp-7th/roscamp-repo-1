from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget
from PyQt6 import QtCore
from PyQt6.QtGui import QMouseEvent


class ClickableLabel(QLabel):
    """마우스 클릭 이벤트를 시그널로 제공하는 QLabel 확장."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """왼쪽 버튼이 떼어질 때 클릭 신호를 방출한다."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)
