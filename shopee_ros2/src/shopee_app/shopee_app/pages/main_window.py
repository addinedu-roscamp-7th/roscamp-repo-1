from typing import TYPE_CHECKING
from typing import Optional

from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QMessageBox

from shopee_app.pages.admin_window import AdminWindow
from shopee_app.pages.user_window import UserWindow
from shopee_app.ui_gen.main_window import Ui_MainWindow

if TYPE_CHECKING:
    from shopee_app.ros_node import RosNodeThread


class MainWindow(QMainWindow):
    def __init__(self, ros_thread: Optional['RosNodeThread'] = None):
        super().__init__()
        self._ros_thread = ros_thread
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

    def on_ros_ready(self):
        # TODO: ROS2 데이터 연동 후 초기 화면 업데이트 로직을 추가한다.
        pass

    def on_ros_error(self, message: str):
        QMessageBox.critical(self, 'ROS2 오류', message)
        self.close()
