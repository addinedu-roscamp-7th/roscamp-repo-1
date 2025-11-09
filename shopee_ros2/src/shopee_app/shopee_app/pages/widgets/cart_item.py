from pathlib import Path
from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QStyle
from PyQt6.QtWidgets import QWidget

from shopee_app.pages.models.cart_item_data import CartItemData
from shopee_app.ui_gen.cart_item import Ui_CartItemWidget


class CartItemWidget(QWidget):
    '''장바구니 항목을 표시하고 수량 조절을 지원하는 위젯.'''

    quantity_changed = pyqtSignal(int, int)
    remove_requested = pyqtSignal(int)
    checked_changed = pyqtSignal(int, bool)

    FALLBACK_IMAGE = (
        Path(__file__).resolve().parent.parent / 'image' / 'product_no_image.png'
    )
    DELETE_ICON = Path(__file__).resolve().parent.parent / 'icons' / 'trash.svg'
    MIN_QUANTITY = 1
    MAX_QUANTITY = 99

    def __init__(self, parent=None):
        '''UI 위젯을 초기화하고 사용자 상호작용 신호를 연결한다.'''
        super().__init__(parent)
        self.ui = Ui_CartItemWidget()
        self.ui.setupUi(self)
        self.configure_delete_button()
        self.data: CartItemData | None = None
        self.ui.btn_decrease.clicked.connect(self.on_decrease_clicked)
        self.ui.btn_increase.clicked.connect(self.on_increase_clicked)
        self.ui.btn_delete.clicked.connect(self.on_delete_clicked)
        self.ui.chk_selected.toggled.connect(self.on_checked_toggled)
        self.ui.edit_quantity.editingFinished.connect(self.on_quantity_edit_finished)

    def apply_item(self, data: CartItemData) -> None:
        '''데이터 모델을 적용하고 UI 요소를 업데이트한다.'''
        self.data = data
        self.ui.label_name.setText(data.name)
        self.ui.edit_quantity.setText(str(data.quantity))
        self.ui.chk_selected.setChecked(data.is_selected)
        self.ui.label_price.setText(f'단가 {data.price:,} 원')
        self.ui.label_total_price.setText(f'{data.total_price:,} 원')
        self.apply_image(data.image_path)

    def set_checked(self, checked: bool) -> None:
        '''체크박스 상태를 외부 제어에 맞춰 반영한다.'''
        self.ui.chk_selected.blockSignals(True)
        self.ui.chk_selected.setChecked(checked)
        self.ui.chk_selected.blockSignals(False)
        if self.data is not None:
            self.data.is_selected = checked

    def apply_image(self, path: Path | None) -> None:
        '''항목 이미지가 없을 경우 대체 이미지를 설정한다.'''
        image_path = path if path and path.exists() else self.FALLBACK_IMAGE
        pixmap = QtGui.QPixmap(str(image_path))
        self.ui.label_image.setPixmap(pixmap)

    def configure_delete_button(self) -> None:
        '''삭제 버튼에 아이콘과 접근성 정보를 설정한다.'''
        if self.DELETE_ICON.exists():
            icon = QtGui.QIcon(str(self.DELETE_ICON))
        else:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        if icon.isNull():
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        self.ui.btn_delete.setIcon(icon)
        self.ui.btn_delete.setIconSize(QtCore.QSize(20, 20))
        self.ui.btn_delete.setText('')
        self.ui.btn_delete.setToolTip('삭제')
        self.ui.btn_delete.setAccessibleName('장바구니 항목 삭제 버튼')

    def on_decrease_clicked(self) -> None:
        '''수량 감소 버튼 클릭 시 수량을 한 단위 줄인다.'''
        if self.data is None:
            return
        quantity = max(self.MIN_QUANTITY, self.current_quantity() - 1)
        self.ui.edit_quantity.setText(str(quantity))
        self.emit_quantity_changed(quantity)

    def on_increase_clicked(self) -> None:
        '''수량 증가 버튼 클릭 시 수량을 한 단위 늘린다.'''
        if self.data is None:
            return
        quantity = min(self.MAX_QUANTITY, self.current_quantity() + 1)
        self.ui.edit_quantity.setText(str(quantity))
        self.emit_quantity_changed(quantity)

    def on_delete_clicked(self) -> None:
        '''삭제 버튼 클릭 시 항목 제거 요청을 보낸다.'''
        if self.data is None:
            return
        self.remove_requested.emit(self.data.product_id)

    def on_checked_toggled(self, checked: bool) -> None:
        '''체크박스 토글에 따라 선택 여부를 알린다.'''
        if self.data is None:
            return
        self.checked_changed.emit(self.data.product_id, checked)

    def on_quantity_edit_finished(self) -> None:
        '''수량 입력 후 포커스가 빠질 때 유효 범위로 값을 정규화한다.'''
        if self.data is None:
            return
        quantity = self.current_quantity()
        quantity = max(self.MIN_QUANTITY, min(self.MAX_QUANTITY, quantity))
        self.ui.edit_quantity.setText(str(quantity))
        self.emit_quantity_changed(quantity)

    def emit_quantity_changed(self, quantity: int) -> None:
        '''수량 변경을 신호로 내보내고 합계를 갱신한다.'''
        if self.data is None:
            return
        if quantity == self.data.quantity:
            self.ui.label_total_price.setText(f'{self.data.total_price:,} 원')
            return
        self.data.quantity = quantity
        self.ui.label_total_price.setText(f'{self.data.total_price:,} 원')
        self.quantity_changed.emit(self.data.product_id, quantity)

    def current_quantity(self) -> int:
        '''수량 입력값을 정수로 변환해 반환한다.'''
        try:
            return int(self.ui.edit_quantity.text())
        except ValueError:
            return self.data.quantity if self.data else self.MIN_QUANTITY
