from pathlib import Path
from typing import Any
from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QWidget

from shopee_app.pages.models.product_data import ProductData
from shopee_app.utils.allergy_utils import get_matching_allergies, get_vegan_status
from shopee_app.ui_gen.promoded_class import Ui_product_form as Ui_PromotionCard


class ProductCard(QWidget):
    FALLBACK_IMAGE = (
        Path(__file__).resolve().parent.parent / "image" / "product_no_image.png"
    )
    DEFAULT_SIZE = QtCore.QSize(230, 315)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_PromotionCard()
        self.ui.setupUi(self)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        button = getattr(self.ui, "btn_add_product", None)
        if button is None:
            button = getattr(self.ui, "toolButton", None)
        icon_path = Path(__file__).resolve().parent.parent / "icons" / "product_add.svg"
        if button is not None:
            if icon_path.exists():
                button.setIcon(QtGui.QIcon(str(icon_path)))
            button.setIconSize(QtCore.QSize(14, 14))
            button.setToolButtonStyle(
                QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            button.setMinimumHeight(26)
            button.setMaximumHeight(26)
            button.setStyleSheet(
                "QToolButton {background-color: transparent; border: none; "
                "color: #FF3134; padding: 0 6px;}"
                "QToolButton:pressed {color: #cc2829;}"
            )
            button.setSizePolicy(
                QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            )
        self.setMinimumSize(self.DEFAULT_SIZE)
        self.setMaximumSize(self.DEFAULT_SIZE)
        size_policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setSizePolicy(size_policy)
        self.ui.label_prod_image.setScaledContents(False)
        self.source_pixmap: QtGui.QPixmap | None = None

    def sizeHint(self) -> QtCore.QSize:
        return self.DEFAULT_SIZE

    def apply_product(
        self, product: ProductData, user_allergy: dict[str, bool] | None = None
    ) -> None:
        self.ui.label_prod_name.setText(product.name)
        self.ui.label_category.setText(product.category)

        # 알러지 정보 표시 (사용자 알러지와 매칭)
        if user_allergy is not None and product.allergy_info is not None:
            # AllergyInfoData를 dict로 변환
            product_allergy = {
                "nuts": product.allergy_info.nuts,
                "milk": product.allergy_info.milk,
                "seafood": product.allergy_info.seafood,
                "soy": product.allergy_info.soy,
                "peach": product.allergy_info.peach,
                "gluten": product.allergy_info.gluten,
                "eggs": product.allergy_info.eggs,
            }
            matched_allergies = get_matching_allergies(user_allergy, product_allergy)
            if matched_allergies:
                self.ui.label_allergy_info.setText(f"⚠️ {matched_allergies}")
            else:
                self.ui.label_allergy_info.setText("안전")
        else:
            self.ui.label_allergy_info.setText("정보 없음")

        # 비건 정보 표시
        self.ui.label_vegan_info.setText(get_vegan_status(product.is_vegan_friendly))

        # 가격 정보 표시
        self.ui.label_original_price.setText(f"{product.price:,} 원")
        self.ui.label_discount_rate.setText(f"{product.discount_rate}%")
        self.ui.label_discounted_price.setText(f"{product.discounted_price:,} 원")
        self.apply_image(product.image_path)

    def apply_image(self, path: Path | None) -> None:
        if path is None or not path.exists():
            path = self.FALLBACK_IMAGE
        self.source_pixmap = QtGui.QPixmap(str(path))
        self.update_image_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_pixmap()

    def update_image_pixmap(self) -> None:
        label = self.ui.label_prod_image
        if self.source_pixmap is None or self.source_pixmap.isNull():
            label.clear()
            return
        target_width = label.width()
        target_height = label.height()
        if target_width <= 0 or target_height <= 0:
            label.setPixmap(self.source_pixmap)
            return
        scaled_pixmap = self.source_pixmap.scaled(
            label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled_pixmap)
