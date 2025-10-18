from pathlib import Path
from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QWidget

from ui_gen.promoded_class import Ui_product_form as Ui_PromotionCard
from pages.models.product_data import ProductData


class ProductCard(QWidget):
    _fallback_image = (
        Path(__file__).resolve().parent.parent / 'image' / 'product_no_image.png'
    )
    _default_size = QtCore.QSize(230, 315)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_PromotionCard()
        self.ui.setupUi(self)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumSize(self._default_size)
        size_policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setSizePolicy(size_policy)
        self.ui.label_prod_image.setScaledContents(False)
        self._source_pixmap: QtGui.QPixmap | None = None

    def sizeHint(self) -> QtCore.QSize:
        return self._default_size

    def apply_product(self, product: ProductData) -> None:
        self.ui.label_prod_name.setText(product.name)
        self.ui.label_category.setText(product.category)
        self.ui.label_allergy_info.setText(f'알레르기 ID: {product.allergy_info_id}')
        vegan_text = '비건 가능' if product.is_vegan_friendly else '비건 불가'
        self.ui.label_vegan_info.setText(vegan_text)
        self.ui.label_original_price.setText(f'{product.price:,} 원')
        self.ui.label_discount_rate.setText(f'{product.discount_rate}%')
        self.ui.label_discounted_price.setText(f'{product.discounted_price:,} 원')
        self.apply_image(product.image_path)

    def apply_image(self, path: Path | None) -> None:
        if path is None or not path.exists():
            path = self._fallback_image
        self._source_pixmap = QtGui.QPixmap(str(path))
        self._update_image_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image_pixmap()

    def _update_image_pixmap(self) -> None:
        label = self.ui.label_prod_image
        if self._source_pixmap is None or self._source_pixmap.isNull():
            label.clear()
            return
        target_width = label.width()
        target_height = label.height()
        if target_width <= 0 or target_height <= 0:
            label.setPixmap(self._source_pixmap)
            return
        scaled_pixmap = self._source_pixmap.scaled(
            label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled_pixmap)
