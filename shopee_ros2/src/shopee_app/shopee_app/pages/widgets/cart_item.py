from pathlib import Path
from PyQt6 import QtGui
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from shopee_app.pages.models.cart_item_data import CartItemData
from shopee_app.ui_gen.cart_item import Ui_CartItemWidget


class CartItemWidget(QWidget):
    quantity_changed = pyqtSignal(int, int)
    remove_requested = pyqtSignal(int)
    checked_changed = pyqtSignal(int, bool)

    _fallback_image = (
        Path(__file__).resolve().parent.parent / 'image' / 'product_no_image.png'
    )
    _min_quantity = 1
    _max_quantity = 99

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_CartItemWidget()
        self.ui.setupUi(self)
        self._data: CartItemData | None = None
        self.ui.btn_decrease.clicked.connect(self._on_decrease_clicked)
        self.ui.btn_increase.clicked.connect(self._on_increase_clicked)
        self.ui.btn_delete.clicked.connect(self._on_delete_clicked)
        self.ui.chk_selected.toggled.connect(self._on_checked_toggled)
        self.ui.edit_quantity.editingFinished.connect(self._on_quantity_edit_finished)

    def apply_item(self, data: CartItemData) -> None:
        self._data = data
        self.ui.label_name.setText(data.name)
        self.ui.edit_quantity.setText(str(data.quantity))
        self.ui.chk_selected.setChecked(data.is_selected)
        self.ui.label_price.setText(f'단가 {data.price:,} 원')
        self.ui.label_total_price.setText(f'{data.total_price:,} 원')
        self._apply_image(data.image_path)

    def set_checked(self, checked: bool) -> None:
        self.ui.chk_selected.blockSignals(True)
        self.ui.chk_selected.setChecked(checked)
        self.ui.chk_selected.blockSignals(False)
        if self._data is not None:
            self._data.is_selected = checked

    def _apply_image(self, path: Path | None) -> None:
        image_path = path if path and path.exists() else self._fallback_image
        pixmap = QtGui.QPixmap(str(image_path))
        self.ui.label_image.setPixmap(pixmap)

    def _on_decrease_clicked(self) -> None:
        if self._data is None:
            return
        quantity = max(self._min_quantity, self._current_quantity() - 1)
        self.ui.edit_quantity.setText(str(quantity))
        self._emit_quantity_changed(quantity)

    def _on_increase_clicked(self) -> None:
        if self._data is None:
            return
        quantity = min(self._max_quantity, self._current_quantity() + 1)
        self.ui.edit_quantity.setText(str(quantity))
        self._emit_quantity_changed(quantity)

    def _on_delete_clicked(self) -> None:
        if self._data is None:
            return
        self.remove_requested.emit(self._data.product_id)

    def _on_checked_toggled(self, checked: bool) -> None:
        if self._data is None:
            return
        self.checked_changed.emit(self._data.product_id, checked)

    def _on_quantity_edit_finished(self) -> None:
        if self._data is None:
            return
        quantity = self._current_quantity()
        quantity = max(self._min_quantity, min(self._max_quantity, quantity))
        self.ui.edit_quantity.setText(str(quantity))
        self._emit_quantity_changed(quantity)

    def _emit_quantity_changed(self, quantity: int) -> None:
        if self._data is None:
            return
        if quantity == self._data.quantity:
            self.ui.label_total_price.setText(f'{self._data.total_price:,} 원')
            return
        self._data.quantity = quantity
        self.ui.label_total_price.setText(f'{self._data.total_price:,} 원')
        self.quantity_changed.emit(self._data.product_id, quantity)

    def _current_quantity(self) -> int:
        try:
            return int(self.ui.edit_quantity.text())
        except ValueError:
            return self._data.quantity if self._data else self._min_quantity
