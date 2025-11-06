from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from shopee_app.ui_gen.layout_admin import Ui_Form_admin as Ui_AdminLayout


class AdminWindow(QWidget):
    '''관리자 전용 간단한 메뉴를 제공하는 윈도우.'''

    closed = pyqtSignal()

    def __init__(self, parent=None):
        '''UI를 초기화하고 로그인 화면으로 돌아가는 버튼을 연결한다.'''
        super().__init__(parent)
        self.ui = Ui_AdminLayout()
        self.ui.setupUi(self)
        self.setWindowTitle('Shopee GUI - Admin')
        self.ui.btn_to_login.clicked.connect(self.close)

    def closeEvent(self, event):
        '''윈도우가 닫힐 때 부모 창에 알림을 전달한다.'''
        self.closed.emit()
        super().closeEvent(event)
