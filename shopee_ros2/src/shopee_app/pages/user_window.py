from pathlib import Path
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtWidgets import QSpacerItem
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QWidget

from ui_gen.layout_user import Ui_Form_user as Ui_UserLayout
from pages.widgets.product_card import ProductCard
from pages.widgets.cart_item import CartItemWidget
from pages.models.product_data import ProductData
from pages.models.cart_item_data import CartItemData


class UserWindow(QWidget):

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_UserLayout()
        self.ui.setupUi(self)

        self.setWindowTitle("Shopee GUI - User")
        self._products_container = getattr(self.ui, "grid_products", None)
        self._product_grid = getattr(self.ui, "gridLayout_2", None)
        self._products: list[ProductData] = []

        self._cart_items: dict[int, CartItemData] = {}
        self._cart_widgets: dict[int, CartItemWidget] = {}
        self._cart_items_layout = getattr(self.ui, "cart_items_layout", None)
        self._cart_spacer = None
        if self._cart_items_layout is not None:
            self._cart_spacer = QSpacerItem(
                0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
            )
            self._cart_items_layout.addItem(self._cart_spacer)
        self.ui.chk_select_all.stateChanged.connect(self._on_select_all_changed)
        if hasattr(self.ui, "btn_selected_delete"):
            self.ui.btn_selected_delete.clicked.connect(
                self._on_delete_selected_clicked
            )

        self._current_columns = 0
        self._cart_container = None
        self._cart_frame = None
        self._cart_body = None
        self._cart_header = None
        self._cart_toggle_button = None
        self._product_scroll = None
        self._cart_expanded = False
        self._cart_container_layout = None
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
        self.setup_cart_section()
        self.setup_navigation()
        self.ui.btn_to_login_page.clicked.connect(self.close)
        self.ui.btn_pay.clicked.connect(self.on_pay_clicked)
        self._products = self.load_initial_products()
        self.refresh_product_grid()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def on_pay_clicked(self):
        self.set_mode("pick")

    def setup_cart_section(self):
        self._cart_container = getattr(self.ui, "widget_3", None)
        self._cart_frame = getattr(self.ui, "cart_frame", None)
        self._cart_body = getattr(self.ui, "cart_body", None)
        self._cart_header = getattr(self.ui, "cart_header", None)
        self._cart_toggle_button = getattr(self.ui, "pushButton_2", None)
        self._product_scroll = getattr(self.ui, "scrollArea", None)

        if not self._cart_frame:
            return

        self._cart_container_layout = (
            self._cart_container.layout() if self._cart_container else None
        )
        if self._cart_container_layout is not None:
            margins = self._cart_container_layout.contentsMargins()
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

        self.apply_cart_state()

    def setup_navigation(self):
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

        self.set_mode("shopping")

    def on_cart_toggle_clicked(self):
        if self._cart_toggle_button is None:
            return

        self._cart_expanded = not self._cart_expanded
        self.apply_cart_state()

    def on_shopping_button_clicked(self):
        self.set_mode("shopping")

    def on_store_button_clicked(self):
        self.set_mode("pick")

    def set_mode(self, mode):
        if mode == self._current_mode:
            return

        self._current_mode = mode

        if mode == "shopping":
            if self._shopping_button:
                self._shopping_button.setChecked(True)
            if self._store_button:
                self._store_button.setChecked(False)
            self.show_main_page(self._page_user)
            self.show_side_page(self._side_pick_filter_page)
            return

        if mode == "pick":
            if self._store_button:
                self._store_button.setChecked(True)
            if self._shopping_button:
                self._shopping_button.setChecked(False)
            self.show_main_page(self._page_pick)
            self.show_side_page(self._side_shop_page)

    def show_main_page(self, page):
        if self._main_stack is None or page is None:
            return
        if self._main_stack.currentWidget() is page:
            return
        self._main_stack.setCurrentWidget(page)

    def show_side_page(self, page):
        if self._side_stack is None or page is None:
            return
        if self._side_stack.currentWidget() is page:
            return
        self._side_stack.setCurrentWidget(page)

    def apply_cart_state(self):
        if self._cart_toggle_button is not None:
            self._cart_toggle_button.setText(
                "접기" if self._cart_expanded else "펼치기"
            )

        if self._cart_body:
            self._cart_body.setVisible(self._cart_expanded)
            if self._cart_expanded:
                self._render_all_cart_items()

        if self._product_scroll is not None:
            self._product_scroll.setVisible(not self._cart_expanded)

        if self._cart_container_layout is not None:
            top_margin = (
                self._cart_margin_expanded_top
                if self._cart_expanded
                else self._cart_margin_collapsed_top
            )
            self._cart_container_layout.setContentsMargins(
                self._cart_margin_left,
                top_margin,
                self._cart_margin_right,
                self._cart_margin_bottom,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_product_grid()

    def refresh_product_grid(self) -> None:
        if self._product_grid is None:
            return

        columns = self.calculate_columns()
        if columns <= 0:
            columns = 1

        if columns == self._current_columns and self._products:
            return

        self.populate_products(self._products, columns)
        self._current_columns = columns

    def calculate_columns(self) -> int:
        scroll_area = getattr(self.ui, "scrollArea", None)
        if scroll_area is not None:
            available_width = scroll_area.viewport().width()
        elif self._products_container is not None:
            available_width = self._products_container.width()
        else:
            available_width = self.width()

        if available_width <= 0:
            return 1

        if self._product_grid is not None:
            margins = self._product_grid.contentsMargins()
            available_width -= margins.left() + margins.right()

        spacing = self._product_grid.horizontalSpacing() if self._product_grid else 0
        if spacing < 0:
            spacing = 0

        card_width = ProductCard._default_size.width()
        total_per_card = card_width + spacing
        if total_per_card <= 0:
            return 1

        columns = max(1, (available_width + spacing) // total_per_card)
        if self._products:
            columns = min(columns, len(self._products))
        return int(columns)

    def populate_products(self, products: list[ProductData], columns: int) -> None:
        if self._product_grid is None:
            return

        while self._product_grid.count():
            item = self._product_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for col in range(columns + 1):
            self._product_grid.setColumnStretch(col, 0)

        for index, product in enumerate(products):
            row = index // columns
            col = index % columns
            card = ProductCard()
            card.apply_product(product)
            if hasattr(card.ui, "toolButton"):
                card.ui.toolButton.clicked.connect(
                    lambda _, p=product: self._on_add_to_cart(p)
                )
            self._product_grid.addWidget(card, row, col)

        rows = (len(products) + columns - 1) // columns if products else 0
        spacer = QSpacerItem(
            0,
            0,
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self._product_grid.addItem(spacer, 0, columns, max(1, rows), 1)
        self._product_grid.setColumnStretch(columns, 1)
        self._product_grid.setHorizontalSpacing(16)
        self._product_grid.setVerticalSpacing(16)
        self._product_grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self._product_grid.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )

    def _on_add_to_cart(self, product: ProductData) -> None:
        item = self._cart_items.get(product.product_id)
        if item is not None:
            item.quantity = min(item.quantity + 1, CartItemWidget._max_quantity)
        else:
            item = CartItemData(
                product_id=product.product_id,
                name=product.name,
                quantity=1,
                price=product.discounted_price,
                image_path=product.image_path,
            )
            self._cart_items[product.product_id] = item
        self._render_cart_item(product.product_id)
        self._update_cart_summary()
        self._sync_select_all_state()

    def _render_all_cart_items(self) -> None:
        if self._cart_items_layout is None:
            return
        for product_id in list(self._cart_widgets.keys()):
            if product_id not in self._cart_items:
                self._remove_cart_widget(product_id)
        for product_id in self._cart_items.keys():
            self._render_cart_item(product_id)
        self._sync_select_all_state()
        self._update_cart_summary()

    def _render_cart_item(self, product_id: int) -> None:
        if self._cart_items_layout is None:
            return
        item = self._cart_items.get(product_id)
        if item is None:
            return
        widget = self._cart_widgets.get(product_id)
        if widget is None:
            widget = CartItemWidget()
            widget.quantity_changed.connect(self._on_item_quantity_changed)
            widget.remove_requested.connect(self._on_item_remove_requested)
            widget.checked_changed.connect(self._on_item_checked_changed)
            insert_index = self._cart_items_layout.count()
            if self._cart_spacer is not None:
                insert_index = max(0, insert_index - 1)
            self._cart_items_layout.insertWidget(insert_index, widget)
            self._cart_widgets[product_id] = widget
        widget.apply_item(item)

    def _remove_cart_widget(self, product_id: int) -> None:
        widget = self._cart_widgets.pop(product_id, None)
        if widget is not None:
            widget.setParent(None)

    def _on_item_remove_requested(self, product_id: int) -> None:
        if product_id in self._cart_items:
            del self._cart_items[product_id]
        self._remove_cart_widget(product_id)
        self._update_cart_summary()
        self._sync_select_all_state()

    def _on_item_quantity_changed(self, product_id: int, quantity: int) -> None:
        item = self._cart_items.get(product_id)
        if item is None:
            return
        item.quantity = quantity
        self._render_cart_item(product_id)
        self._update_cart_summary()

    def _on_item_checked_changed(self, product_id: int, checked: bool) -> None:
        item = self._cart_items.get(product_id)
        if item is None:
            return
        item.is_selected = checked
        self._update_cart_summary()
        self._sync_select_all_state()

    def _on_select_all_changed(self, state: int) -> None:
        if not self._cart_items:
            self.ui.chk_select_all.blockSignals(True)
            self.ui.chk_select_all.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.ui.chk_select_all.blockSignals(False)
            return
        checked = state == QtCore.Qt.CheckState.Checked
        for product_id, item in self._cart_items.items():
            item.is_selected = checked
            widget = self._cart_widgets.get(product_id)
            if widget is None:
                self._render_cart_item(product_id)
                widget = self._cart_widgets.get(product_id)
            if widget is not None:
                widget.set_checked(checked)
        self._update_cart_summary()
        self._sync_select_all_state()

    def _on_delete_selected_clicked(self) -> None:
        for product_id, item in list(self._cart_items.items()):
            if item.is_selected:
                self._on_item_remove_requested(product_id)

    def _sync_select_all_state(self) -> None:
        if not self._cart_items:
            self.ui.chk_select_all.blockSignals(True)
            self.ui.chk_select_all.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.ui.chk_select_all.blockSignals(False)
            return
        selected = sum(1 for item in self._cart_items.values() if item.is_selected)
        state = (
            QtCore.Qt.CheckState.Checked
            if selected == len(self._cart_items)
            else QtCore.Qt.CheckState.Unchecked
        )
        self.ui.chk_select_all.blockSignals(True)
        self.ui.chk_select_all.setCheckState(state)
        self.ui.chk_select_all.blockSignals(False)

    def _update_cart_summary(self) -> None:
        total_qty = sum(
            item.quantity for item in self._cart_items.values() if item.is_selected
        )
        total_price = sum(
            item.total_price for item in self._cart_items.values() if item.is_selected
        )
        label_qty = getattr(self.ui, "label_total_qty", None)
        label_price = getattr(self.ui, "label_total_price", None)
        if label_qty is not None:
            label_qty.setText(f"{total_qty}개")
        if label_price is not None:
            label_price.setText(f"{total_price:,} 원")

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
