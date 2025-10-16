from PyQt6 import QtCore
from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPropertyAnimation,
    pyqtSignal,
)
from PyQt6.QtWidgets import QWidget

from ui_gen.layout_user import Ui_Form_user as Ui_UserLayout


class UserWindow(QWidget):

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_UserLayout()
        self.ui.setupUi(self)
        self.setWindowTitle('Shopee GUI - User')
        self._setup_cart_section()
        self.ui.btn_to_login_page.clicked.connect(self.close)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def _setup_cart_section(self):
        '''Initialize cart section toggle and geometry-based animation.'''
        self._cart_container = getattr(self.ui, 'widget_3', None)
        self._cart_frame = getattr(self.ui, 'cart_frame', None)
        self._cart_body = getattr(self.ui, 'cart_body', None)
        self._cart_header = getattr(self.ui, 'cart_header', None)
        self._cart_toggle_button = getattr(self.ui, 'pushButton_2', None)
        self._product_scroll = getattr(self.ui, 'scrollArea', None)

        self._cart_expanded = False
        self._cart_animation = None
        self._cart_collapsed_rect = QtCore.QRect()
        self._cart_expanded_rect = QtCore.QRect()
        self._cart_header_height = 0
        self._cart_body_height = 0

        if self._cart_frame and self._cart_container:
            container_layout = self._cart_container.layout()
            if container_layout is not None:
                container_layout.removeWidget(self._cart_frame)

            self._cart_frame.setParent(self._cart_container)
            self._cart_frame.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_StyledBackground, True
            )
            self._cart_frame.raise_()
            self._cart_frame.show()

            if self._cart_header:
                self._cart_header_height = self._cart_header.sizeHint().height()
                if not self._cart_header_height:
                    self._cart_header_height = (
                        self._cart_header.minimumSizeHint().height()
                    )
            if not self._cart_header_height:
                self._cart_header_height = 64

            if self._cart_body:
                self._cart_body_height = self._cart_body.sizeHint().height()
                if not self._cart_body_height:
                    self._cart_body_height = self._cart_body.minimumSizeHint().height()
                if self._cart_body_height <= 0:
                    self._cart_body_height = 320
                self._cart_body.hide()

            self._cart_animation = QPropertyAnimation(
                self._cart_frame, b'geometry', self
            )
            self._cart_animation.setDuration(280)
            self._cart_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._cart_animation.finished.connect(self._on_cart_animation_finished)

            self._recalculate_cart_geometry()
            self._apply_cart_geometry(force=True)

        if self._product_scroll is not None:
            self._product_scroll.show()

        if self._cart_toggle_button is not None:
            self._cart_toggle_button.setText('펼치기')
            self._cart_toggle_button.clicked.connect(self.on_cart_toggle_clicked)

    def on_cart_toggle_clicked(self):
        if self._cart_toggle_button is None:
            return

        self._cart_expanded = not self._cart_expanded
        self._cart_toggle_button.setText('접기' if self._cart_expanded else '펼치기')

        if self._cart_frame is None:
            if self._cart_body:
                self._cart_body.setVisible(self._cart_expanded)
            if self._product_scroll is not None:
                self._product_scroll.setVisible(not self._cart_expanded)
            return

        self._recalculate_cart_geometry()

        if (
            self._cart_animation
            and self._cart_animation.state() == QAbstractAnimation.State.Running
        ):
            self._cart_animation.stop()

        if self._cart_animation is None:
            self._apply_cart_geometry(force=True)
            return

        if self._cart_expanded:
            if self._product_scroll is not None:
                self._product_scroll.hide()
            if self._cart_body:
                self._cart_body.show()
        else:
            if self._cart_body:
                self._cart_body.show()

        start_rect = self._cart_frame.geometry()
        end_rect = (
            self._cart_expanded_rect
            if self._cart_expanded
            else self._cart_collapsed_rect
        )
        self._cart_animation.setStartValue(start_rect)
        self._cart_animation.setEndValue(end_rect)
        self._cart_animation.start()

    def _on_cart_animation_finished(self):
        if self._cart_frame is None:
            return

        if self._cart_expanded:
            self._cart_frame.setGeometry(self._cart_expanded_rect)
            if self._cart_body:
                self._cart_body.show()
        else:
            self._cart_frame.setGeometry(self._cart_collapsed_rect)
            if self._cart_body:
                self._cart_body.hide()
            if self._product_scroll is not None:
                self._product_scroll.show()

    def _recalculate_cart_geometry(self):
        if self._cart_frame is None or self._cart_container is None:
            return

        host_rect = self._cart_container.rect()
        base_rect = (
            self._product_scroll.geometry()
            if self._product_scroll is not None and not self._product_scroll.isHidden()
            else QtCore.QRect(host_rect)
        )

        width = base_rect.width()
        left = base_rect.left()
        top = base_rect.top()
        bottom = base_rect.top() + base_rect.height()

        collapsed_height = self._cart_header_height
        expanded_height = collapsed_height + self._cart_body_height
        collapsed_top = bottom - collapsed_height
        expanded_top = max(top, bottom - expanded_height)

        self._cart_collapsed_rect = QtCore.QRect(
            left, collapsed_top, width, collapsed_height
        )
        self._cart_expanded_rect = QtCore.QRect(
            left, expanded_top, width, expanded_height
        )

    def _apply_cart_geometry(self, force=False):
        if self._cart_frame is None:
            return

        if (
            not force
            and self._cart_animation
            and self._cart_animation.state() == QAbstractAnimation.State.Running
        ):
            return

        target_rect = (
            self._cart_expanded_rect
            if self._cart_expanded
            else self._cart_collapsed_rect
        )
        if target_rect.isValid():
            self._cart_frame.setGeometry(target_rect)

        if self._cart_body:
            self._cart_body.setVisible(self._cart_expanded)

        if self._product_scroll is not None:
            self._product_scroll.setVisible(not self._cart_expanded)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recalculate_cart_geometry()
        if (
            self._cart_animation
            and self._cart_animation.state() == QAbstractAnimation.State.Running
        ):
            self._cart_animation.stop()
        self._apply_cart_geometry(force=True)
