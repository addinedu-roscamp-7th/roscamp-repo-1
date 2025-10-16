from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtWidgets import QWidget

from ui_gen.layout_user import Ui_Form_user as Ui_UserLayout


class UserWindow(QWidget):

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_UserLayout()
        self.ui.setupUi(self)
        self.setWindowTitle("Shopee GUI - User")
        self._cart_container = None
        self._cart_frame = None
        self._cart_body = None
        self._cart_header = None
        self._cart_toggle_button = None
        self._product_scroll = None
        self._cart_expanded = False
        self._cart_layout = None
        self._cart_margin_left = 0
        self._cart_margin_right = 0
        self._cart_margin_bottom = 0
        self._cart_margin_collapsed_top = 0
        self._cart_margin_expanded_top = 24
        self._main_stack = None
        self._side_stack = None
        self._page_user = None
        self._page_pick = None
        self._side_shop_page = None
        self._side_pick_filter_page = None
        self._shopping_button = None
        self._store_button = None
        self._nav_group = None
        self._current_mode = None
        self._setup_cart_section()
        self._setup_navigation()
        self.ui.btn_to_login_page.clicked.connect(self.close)
        self.ui.btn_pay.clicked.connect(self._on_pay_clicked)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def _on_pay_clicked(self):
        self._set_mode("pick")

    def _setup_cart_section(self):
        self._cart_container = getattr(self.ui, "widget_3", None)
        self._cart_frame = getattr(self.ui, "cart_frame", None)
        self._cart_body = getattr(self.ui, "cart_body", None)
        self._cart_header = getattr(self.ui, "cart_header", None)
        self._cart_toggle_button = getattr(self.ui, "pushButton_2", None)
        self._product_scroll = getattr(self.ui, "scrollArea", None)

        if not self._cart_frame:
            return

        self._cart_layout = (
            self._cart_container.layout() if self._cart_container else None
        )
        if self._cart_layout is not None:
            margins = self._cart_layout.contentsMargins()
            self._cart_margin_left = margins.left()
            self._cart_margin_right = margins.right()
            self._cart_margin_bottom = margins.bottom()
            self._cart_margin_collapsed_top = margins.top()
            self._cart_margin_expanded_top = self._cart_margin_collapsed_top + max(
                8, self._cart_margin_expanded_top
            )

        self._cart_frame.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )

        if self._cart_body:
            self._cart_body.hide()

        if self._cart_toggle_button is not None:
            self._cart_toggle_button.setText("펼치기")
            self._cart_toggle_button.clicked.connect(self.on_cart_toggle_clicked)

        if self._product_scroll is not None:
            self._product_scroll.show()

        self._apply_cart_state()

    def _setup_navigation(self):
        self._main_stack = getattr(self.ui, "stacked_content", None)
        self._side_stack = getattr(self.ui, "stack_side_bar", None)
        self._page_user = getattr(self.ui, "page_content_user", None)
        self._page_pick = getattr(self.ui, "page_content_pick", None)
        self._side_shop_page = getattr(self.ui, "side_pick_page", None)
        self._side_pick_filter_page = getattr(self.ui, "side_allergy_filter_page", None)
        self._shopping_button = getattr(self.ui, "toolButton_3", None)
        self._store_button = getattr(self.ui, "toolButton_2", None)

        if self._shopping_button:
            self._shopping_button.setCheckable(True)
        if self._store_button:
            self._store_button.setCheckable(True)

        if self._shopping_button or self._store_button:
            self._nav_group = QButtonGroup(self)
            self._nav_group.setExclusive(True)
            if self._shopping_button:
                self._nav_group.addButton(self._shopping_button)
            if self._store_button:
                self._nav_group.addButton(self._store_button)

        if self._shopping_button:
            self._shopping_button.clicked.connect(self.on_shopping_button_clicked)
        if self._store_button:
            self._store_button.clicked.connect(self.on_store_button_clicked)

        self._set_mode("shopping")

    def on_cart_toggle_clicked(self):
        if self._cart_toggle_button is None:
            return

        self._cart_expanded = not self._cart_expanded
        self._apply_cart_state()

    def on_shopping_button_clicked(self):
        self._set_mode("shopping")

    def on_store_button_clicked(self):
        self._set_mode("pick")

    def _set_mode(self, mode):
        if mode == self._current_mode:
            return

        self._current_mode = mode

        if mode == "shopping":
            if self._shopping_button:
                self._shopping_button.setChecked(True)
            if self._store_button:
                self._store_button.setChecked(False)
            self._show_main_page(self._page_user)
            self._show_side_page(self._side_pick_filter_page)
            return

        if mode == "pick":
            if self._store_button:
                self._store_button.setChecked(True)
            if self._shopping_button:
                self._shopping_button.setChecked(False)
            self._show_main_page(self._page_pick)
            self._show_side_page(self._side_shop_page)

    def _show_main_page(self, page):
        if self._main_stack is None or page is None:
            return
        if self._main_stack.currentWidget() is page:
            return
        self._main_stack.setCurrentWidget(page)

    def _show_side_page(self, page):
        if self._side_stack is None or page is None:
            return
        if self._side_stack.currentWidget() is page:
            return
        self._side_stack.setCurrentWidget(page)

    def _apply_cart_state(self):
        if self._cart_toggle_button is not None:
            self._cart_toggle_button.setText(
                "접기" if self._cart_expanded else "펼치기"
            )

        if self._cart_body:
            self._cart_body.setVisible(self._cart_expanded)

        if self._product_scroll is not None:
            self._product_scroll.setVisible(not self._cart_expanded)

        if self._cart_layout is not None:
            top_margin = (
                self._cart_margin_expanded_top
                if self._cart_expanded
                else self._cart_margin_collapsed_top
            )
            self._cart_layout.setContentsMargins(
                self._cart_margin_left,
                top_margin,
                self._cart_margin_right,
                self._cart_margin_bottom,
            )
