from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QSpacerItem
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QWidget
from pathlib import Path

from shopee_app.services.main_service_client import MainServiceClient
from shopee_app.services.main_service_client import MainServiceClientError
from shopee_app.pages.models.cart_item_data import CartItemData
from shopee_app.pages.models.product_data import ProductData
from shopee_app.pages.widgets.cart_item import CartItemWidget
from shopee_app.pages.widgets.product_card import ProductCard
from shopee_app.ui_gen.layout_user import Ui_Form_user as Ui_UserLayout
from shopee_app.pages.widgets.profile_dialog import ProfileDialog


class UserWindow(QWidget):

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_UserLayout()
        self.ui.setupUi(self)

        self.setWindowTitle("Shopee GUI - User")
        self.products_container = getattr(self.ui, "grid_products", None)
        self.product_grid = getattr(self.ui, "gridLayout_2", None)
        self.products: list[ProductData] = []

        self.cart_items: dict[int, CartItemData] = {}
        self.cart_widgets: dict[int, CartItemWidget] = {}
        self.cart_items_layout = getattr(self.ui, "cart_items_layout", None)
        self.cart_spacer = None
        if self.cart_items_layout is not None:
            self.cart_spacer = QSpacerItem(
                0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
            )
            self.cart_items_layout.addItem(self.cart_spacer)
        self.ui.chk_select_all.stateChanged.connect(self.on_select_all_changed)
        if hasattr(self.ui, "btn_selected_delete"):
            self.ui.btn_selected_delete.clicked.connect(self.on_delete_selected_clicked)

        self.current_columns = -1
        self.cart_container = None
        self.cart_frame = None
        self.cart_body = None
        self.cart_header = None
        self.cart_toggle_button = None
        self.product_scroll = None
        self.cart_expanded = False
        self.cart_container_layout = None
        self.cart_margin_left = 0
        self.cart_margin_right = 0
        self.cart_margin_bottom = 0
        self.cart_margin_collapsed_top = 0
        self.cart_margin_expanded_top = 24
        self.main_stack = None
        self.side_stack = None
        self.page_user = None
        self.page_pick = None
        self.side_shop_page = None
        self.side_pick_filter_page = None
        self.shopping_button = None
        self.store_button = None
        self.nav_group = None
        self.current_mode = None
        self.setup_cart_section()
        self.setup_navigation()
        self.ui.btn_to_login_page.clicked.connect(self.close)
        self.ui.btn_pay.clicked.connect(self.on_pay_clicked)
        self.products = self.load_initial_products()
        self.update_cart_summary()

        self.service_client = MainServiceClient()
        self.current_user_id = "admin"

        self.profile_dialog = ProfileDialog(self)
        self.ui.btn_profile.clicked.connect(self.open_profile_dialog)
        QtCore.QTimer.singleShot(0, self.refresh_product_grid)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def on_pay_clicked(self):
        if self.request_create_order():
            self.set_mode("pick")

    def setup_cart_section(self):
        self.cart_container = getattr(self.ui, "widget_3", None)
        self.cart_frame = getattr(self.ui, "cart_frame", None)
        self.cart_body = getattr(self.ui, "cart_body", None)
        self.cart_header = getattr(self.ui, "cart_header", None)
        self.cart_toggle_button = getattr(self.ui, "pushButton_2", None)
        self.product_scroll = getattr(self.ui, "scrollArea", None)

        if not self.cart_frame:
            return

        self.cart_container_layout = (
            self.cart_container.layout() if self.cart_container else None
        )
        if self.cart_container_layout is not None:
            margins = self.cart_container_layout.contentsMargins()
            self.cart_margin_left = margins.left()
            self.cart_margin_right = margins.right()
            self.cart_margin_bottom = margins.bottom()
            self.cart_margin_collapsed_top = margins.top()
            self.cart_margin_expanded_top = self.cart_margin_collapsed_top + max(
                8, self.cart_margin_expanded_top
            )

        self.cart_frame.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )

        if self.cart_body:
            self.cart_body.hide()

        if self.cart_toggle_button is not None:
            self.cart_toggle_button.setText("펼치기")
            self.cart_toggle_button.clicked.connect(self.on_cart_toggle_clicked)

        if self.product_scroll is not None:
            self.product_scroll.show()

        self.apply_cart_state()

    def setup_navigation(self):
        self.main_stack = getattr(self.ui, "stacked_content", None)
        self.side_stack = getattr(self.ui, "stack_side_bar", None)
        self.page_user = getattr(self.ui, "page_content_user", None)
        self.page_pick = getattr(self.ui, "page_content_pick", None)
        self.side_shop_page = getattr(self.ui, "side_pick_page", None)
        self.side_pick_filter_page = getattr(self.ui, "side_allergy_filter_page", None)
        self.shopping_button = getattr(self.ui, "toolButton_3", None)
        self.store_button = getattr(self.ui, "toolButton_2", None)

        if self.shopping_button:
            self.shopping_button.setCheckable(True)
        if self.store_button:
            self.store_button.setCheckable(True)

        if self.shopping_button or self.store_button:
            self.nav_group = QButtonGroup(self)
            self.nav_group.setExclusive(True)
            if self.shopping_button:
                self.nav_group.addButton(self.shopping_button)
            if self.store_button:
                self.nav_group.addButton(self.store_button)

        if self.shopping_button:
            self.shopping_button.clicked.connect(self.on_shopping_button_clicked)
        if self.store_button:
            self.store_button.clicked.connect(self.on_store_button_clicked)

        self.set_mode("shopping")

    def on_cart_toggle_clicked(self):
        if self.cart_toggle_button is None:
            return

        self.cart_expanded = not self.cart_expanded
        self.apply_cart_state()

    def on_shopping_button_clicked(self):
        self.set_mode("shopping")

    def on_store_button_clicked(self):
        self.set_mode("pick")

    def request_create_order(self) -> bool:
        if not getattr(self, "service_client", None):
            return False

        user_id = getattr(self, "current_user_id", "")
        if not user_id:
            QMessageBox.warning(self, "주문 생성 실패", "사용자 정보가 없습니다.")
            return False

        selected_items = [item for item in self.cart_items.values() if item.is_selected]
        if not selected_items:
            QMessageBox.warning(self, "주문 생성 실패", "선택된 상품이 없습니다.")
            return False

        total_amount = sum(item.total_price for item in selected_items)
        payment_method = "card"

        try:
            response = self.service_client.create_order(
                user_id=user_id,
                cart_items=selected_items,
                payment_method=payment_method,
                total_amount=total_amount,
            )
        except MainServiceClientError as exc:
            QMessageBox.warning(self, "주문 생성 실패", str(exc))
            return False

        if not response:
            QMessageBox.warning(self, "주문 생성 실패", "서버 응답이 없습니다.")
            return False

        if response.get("result"):
            QMessageBox.information(
                self,
                "주문 생성 완료",
                response.get("message") or "주문이 생성되었습니다.",
            )
            return True

        QMessageBox.warning(
            self,
            "주문 생성 실패",
            response.get("message") or "주문 생성에 실패했습니다.",
        )
        return False

    def open_profile_dialog(self) -> None:
        dialog = getattr(self, "profile_dialog", None)
        button = getattr(self.ui, "btn_profile", None)
        if dialog is None or button is None:
            return

        dialog.adjustSize()
        anchor = button.mapToGlobal(button.rect().bottomRight())
        x = anchor.x() - dialog.width()
        y = anchor.y() + 6

        parent_widget = self.window()
        if parent_widget is not None:
            parent_top_left = parent_widget.mapToGlobal(QtCore.QPoint(0, 0))
            parent_rect = QtCore.QRect(parent_top_left, parent_widget.size())
            if x < parent_rect.left():
                x = parent_rect.left()
            if x + dialog.width() > parent_rect.right():
                x = parent_rect.right() - dialog.width()
            if y + dialog.height() > parent_rect.bottom():
                y = anchor.y() - dialog.height() - 6
            if y < parent_rect.top():
                y = parent_rect.top()
        else:
            window_handle = button.window().windowHandle() if button.window() else None
            screen = window_handle.screen() if window_handle is not None else None
            if screen is not None:
                available = screen.availableGeometry()
                if x < available.left():
                    x = available.left()
                if x + dialog.width() > available.right():
                    x = available.right() - dialog.width()
                if y + dialog.height() > available.bottom():
                    y = anchor.y() - dialog.height() - 6
                if y < available.top():
                    y = available.top()

        dialog.move(int(x), int(y))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def set_mode(self, mode):
        if mode == self.current_mode:
            return

        self.current_mode = mode

        if mode == "shopping":
            if self.shopping_button:
                self.shopping_button.setChecked(True)
            if self.store_button:
                self.store_button.setChecked(False)
            self.show_main_page(self.page_user)
            self.show_side_page(self.side_pick_filter_page)
            return

        if mode == "pick":
            if self.store_button:
                self.store_button.setChecked(True)
            if self.shopping_button:
                self.shopping_button.setChecked(False)
            self.show_main_page(self.page_pick)
            self.show_side_page(self.side_shop_page)

    def show_main_page(self, page):
        if self.main_stack is None or page is None:
            return
        if self.main_stack.currentWidget() is page:
            return
        self.main_stack.setCurrentWidget(page)

    def show_side_page(self, page):
        if self.side_stack is None or page is None:
            return
        if self.side_stack.currentWidget() is page:
            return
        self.side_stack.setCurrentWidget(page)

    def apply_cart_state(self):
        if self.cart_toggle_button is not None:
            self.cart_toggle_button.setText("접기" if self.cart_expanded else "펼치기")

        if self.cart_body:
            self.cart_body.setVisible(self.cart_expanded)
            if self.cart_expanded:
                self.render_all_cart_items()

        if self.product_scroll is not None:
            self.product_scroll.setVisible(not self.cart_expanded)

        if self.cart_container_layout is not None:
            top_margin = (
                self.cart_margin_expanded_top
                if self.cart_expanded
                else self.cart_margin_collapsed_top
            )
            self.cart_container_layout.setContentsMargins(
                self.cart_margin_left,
                top_margin,
                self.cart_margin_right,
                self.cart_margin_bottom,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_product_grid()

    def refresh_product_grid(self) -> None:
        if self.product_grid is None:
            return

        columns = self.calculate_columns()
        if columns <= 0:
            columns = 1

        if columns == self.current_columns and self.products:
            return

        self.populate_products(self.products, columns)
        self.current_columns = columns

    def calculate_columns(self) -> int:
        scroll_area = getattr(self.ui, "scrollArea", None)
        if scroll_area is not None:
            available_width = scroll_area.viewport().width()
        elif self.products_container is not None:
            available_width = self.products_container.width()
        else:
            available_width = self.width()

        if available_width <= 0:
            return 1

        if self.product_grid is not None:
            margins = self.product_grid.contentsMargins()
            available_width -= margins.left() + margins.right()

        spacing = self.product_grid.horizontalSpacing() if self.product_grid else 0
        if spacing < 0:
            spacing = 0

        card_width = ProductCard.DEFAULT_SIZE.width()
        total_per_card = card_width + spacing
        if total_per_card <= 0:
            return 1

        columns = max(1, (available_width + spacing) // total_per_card)
        if self.products:
            columns = min(columns, len(self.products))
        return int(columns)

    def populate_products(
        self,
        products: list[ProductData],
        columns: int,
    ) -> None:
        if self.product_grid is None:
            return

        while self.product_grid.count():
            item = self.product_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for col in range(columns + 1):
            self.product_grid.setColumnStretch(col, 0)

        for index, product in enumerate(products):
            row = index // columns
            col = index % columns
            card = ProductCard()
            card.apply_product(product)
            if hasattr(card.ui, "toolButton"):
                card.ui.toolButton.clicked.connect(
                    lambda _, p=product: self.on_add_to_cart(p)
                )
            self.product_grid.addWidget(card, row, col)

        rows = (len(products) + columns - 1) // columns if products else 0
        spacer = QSpacerItem(
            0,
            0,
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.product_grid.addItem(spacer, 0, columns, max(1, rows), 1)
        self.product_grid.setColumnStretch(columns, 1)
        self.product_grid.setHorizontalSpacing(16)
        self.product_grid.setVerticalSpacing(16)
        self.product_grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.product_grid.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )

    def on_add_to_cart(self, product: ProductData) -> None:
        item = self.cart_items.get(product.product_id)
        if item is not None:
            item.quantity = min(item.quantity + 1, CartItemWidget.MAX_QUANTITY)
        else:
            item = CartItemData(
                product_id=product.product_id,
                name=product.name,
                quantity=1,
                price=product.discounted_price,
                image_path=product.image_path,
            )
            self.cart_items[product.product_id] = item
        self.render_cart_item(product.product_id)
        self.update_cart_summary()
        self.sync_select_all_state()

    def render_all_cart_items(self) -> None:
        if self.cart_items_layout is None:
            return
        for product_id in list(self.cart_widgets.keys()):
            if product_id not in self.cart_items:
                self.remove_cart_widget(product_id)
        for product_id in self.cart_items.keys():
            self.render_cart_item(product_id)
        self.sync_select_all_state()
        self.update_cart_summary()

    def render_cart_item(self, product_id: int) -> None:
        if self.cart_items_layout is None:
            return
        item = self.cart_items.get(product_id)
        if item is None:
            return
        widget = self.cart_widgets.get(product_id)
        if widget is None:
            widget = CartItemWidget()
            widget.quantity_changed.connect(self.on_item_quantity_changed)
            widget.remove_requested.connect(self.on_item_remove_requested)
            widget.checked_changed.connect(self.on_item_checked_changed)
            insert_index = self.cart_items_layout.count()
            if self.cart_spacer is not None:
                insert_index = max(0, insert_index - 1)
            self.cart_items_layout.insertWidget(insert_index, widget)
            self.cart_widgets[product_id] = widget
        widget.apply_item(item)

    def remove_cart_widget(self, product_id: int) -> None:
        widget = self.cart_widgets.pop(product_id, None)
        if widget is not None:
            widget.setParent(None)

    def on_item_remove_requested(self, product_id: int) -> None:
        if product_id in self.cart_items:
            del self.cart_items[product_id]
        self.remove_cart_widget(product_id)
        self.update_cart_summary()
        self.sync_select_all_state()

    def on_item_quantity_changed(self, product_id: int, quantity: int) -> None:
        item = self.cart_items.get(product_id)
        if item is None:
            return
        item.quantity = quantity
        self.render_cart_item(product_id)
        self.update_cart_summary()

    def on_item_checked_changed(self, product_id: int, checked: bool) -> None:
        item = self.cart_items.get(product_id)
        if item is None:
            return
        item.is_selected = checked
        self.update_cart_summary()
        self.sync_select_all_state()

    def on_select_all_changed(self, state: int) -> None:
        if not self.cart_items:
            self.ui.chk_select_all.blockSignals(True)
            self.ui.chk_select_all.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.ui.chk_select_all.blockSignals(False)
            return
        checked = state == QtCore.Qt.CheckState.Checked
        for product_id, item in self.cart_items.items():
            item.is_selected = checked
            widget = self.cart_widgets.get(product_id)
            if widget is None:
                self.render_cart_item(product_id)
                widget = self.cart_widgets.get(product_id)
            if widget is not None:
                widget.set_checked(checked)
        self.update_cart_summary()
        self.sync_select_all_state()

    def on_delete_selected_clicked(self) -> None:
        for product_id, item in list(self.cart_items.items()):
            if item.is_selected:
                self.on_item_remove_requested(product_id)

    def sync_select_all_state(self) -> None:
        if not self.cart_items:
            self.ui.chk_select_all.blockSignals(True)
            self.ui.chk_select_all.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.ui.chk_select_all.blockSignals(False)
            return
        selected = sum(1 for item in self.cart_items.values() if item.is_selected)
        state = (
            QtCore.Qt.CheckState.Checked
            if selected == len(self.cart_items)
            else QtCore.Qt.CheckState.Unchecked
        )
        self.ui.chk_select_all.blockSignals(True)
        self.ui.chk_select_all.setCheckState(state)
        self.ui.chk_select_all.blockSignals(False)

    def update_cart_summary(self) -> None:
        total_qty = sum(
            item.quantity for item in self.cart_items.values() if item.is_selected
        )
        total_price = sum(
            item.total_price for item in self.cart_items.values() if item.is_selected
        )
        label_quantity = getattr(self.ui, "label_total_count", None)
        label_amount = getattr(self.ui, "label_pay_price", None)
        if label_quantity is not None:
            label_quantity.setText(f"{total_qty}")
        if label_amount is not None:
            label_amount.setText(f"{total_price:,}")

    def load_initial_products(self) -> list[ProductData]:
        image_root = Path(__file__).resolve().parent.parent / "image"
        return [
            ProductData(
                product_id=1,
                name="삼겹살",
                category="고기",
                price=15000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=True,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=2,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=3,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=4,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=2,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=2,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=2,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
        ]
