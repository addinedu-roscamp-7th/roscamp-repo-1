from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from shopee_app.ui_gen.layout_admin import Ui_Form_admin as Ui_AdminLayout


class AdminWindow(QWidget):

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_AdminLayout()
        self.ui.setupUi(self)
        self.setWindowTitle('Shopee GUI - Admin')
        self.ui.btn_to_login.clicked.connect(self.close)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
