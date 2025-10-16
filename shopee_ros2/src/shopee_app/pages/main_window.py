from pathlib import Path
from PyQt6.QtWidgets import QButtonGroup, QMainWindow
from ui_gen.main_window import Ui_MainWindow
from pages.user_window import UserWindow
from pages.admin_window import AdminWindow

ROOT = Path(__file__).resolve().parent.parent  # 프로젝트 루트 경로
UI_DIR = ROOT / 'ui'


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('Shopee GUI (PyQt6)')
        self.setup_role_toggle()
        self.child_window = None
        self.ui.btn_login.clicked.connect(self.on_login_clicked)

    def setup_role_toggle(self):
        self.ui.btn_user.setCheckable(True)
        self.ui.btn_admin.setCheckable(True)
        self.role_group = QButtonGroup(self)
        self.role_group.setExclusive(True)
        self.role_group.addButton(self.ui.btn_user)
        self.role_group.addButton(self.ui.btn_admin)
        self.ui.btn_user.setChecked(True)

    def on_login_clicked(self):
        if self.ui.btn_user.isChecked():
            self.launch_role_window(UserWindow)
            return

        if self.ui.btn_admin.isChecked():
            self.launch_role_window(AdminWindow)
            return

    def launch_role_window(self, window_class):
        if self.child_window:
            self.child_window.close()
            self.child_window = None

        self.child_window = window_class()
        self.child_window.show()
        self.hide()
        self.child_window.closed.connect(self.on_child_closed)

    def on_child_closed(self):
        self.child_window = None
        self.show()
