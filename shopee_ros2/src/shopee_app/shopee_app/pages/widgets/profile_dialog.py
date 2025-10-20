from pathlib import Path

from PyQt6 import QtCore
from PyQt6 import QtWidgets
from PyQt6 import uic


UI_PATH = Path(__file__).resolve().parent.parent.parent / 'ui' / 'dialog_profile.ui'


class ProfileDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self.configure_window()

    def configure_window(self) -> None:
        self.setWindowFlags(
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(260, 220)
