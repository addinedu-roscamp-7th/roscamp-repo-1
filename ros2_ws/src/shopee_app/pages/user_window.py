from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from ui_gen.layout_user import Ui_Form_user as Ui_UserLayout


class UserWindow(QWidget):

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_UserLayout()
        self.ui.setupUi(self)
        self.setWindowTitle('Shopee GUI - User')
        self._cart_container = None
        self._cart_frame = None
        self._cart_body = None
        self._cart_header = None
        self._cart_toggle_button = None
        self._product_scroll = None
        self._cart_expanded = False
        self._setup_cart_section()
        self.ui.btn_to_login_page.clicked.connect(self.close)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def _setup_cart_section(self):
        '''장바구니 토글에 필요한 위젯을 초기화. '''
        self._cart_container = getattr(self.ui, 'widget_3', None)
        self._cart_frame = getattr(self.ui, 'cart_frame', None)
        self._cart_body = getattr(self.ui, 'cart_body', None)
        self._cart_header = getattr(self.ui, 'cart_header', None)
        self._cart_toggle_button = getattr(self.ui, 'pushButton_2', None)
        self._product_scroll = getattr(self.ui, 'scrollArea', None)

        if not self._cart_frame:
            return

        self._cart_frame.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )

        if self._cart_body:
            self._cart_body.hide()

        if self._cart_toggle_button is not None:
            self._cart_toggle_button.setText('펼치기')
            self._cart_toggle_button.clicked.connect(self.on_cart_toggle_clicked)

        if self._product_scroll is not None:
            self._product_scroll.show()

    def on_cart_toggle_clicked(self):
        if self._cart_toggle_button is None:
            return

        self._cart_expanded = not self._cart_expanded
        self._apply_cart_state()

    def _apply_cart_state(self):
        if self._cart_toggle_button is not None:
            self._cart_toggle_button.setText(
                '접기' if self._cart_expanded else '펼치기'
            )

        if self._cart_body:
            self._cart_body.setVisible(self._cart_expanded)

        if self._product_scroll is not None:
            self._product_scroll.setVisible(not self._cart_expanded)
