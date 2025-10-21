from pathlib import Path
from typing import Any

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget


class CartSelectItemWidget(QWidget):
    FALLBACK_IMAGE = (
        Path(__file__).resolve().parent.parent / "image" / "product_no_image.png"
    )
    IMAGE_SIZE = QtCore.QSize(40, 40)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_image = QLabel("", self)
        self.label_image.setFixedSize(self.IMAGE_SIZE)
        size_policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.label_image.setSizePolicy(size_policy)
        self.label_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_image.setScaledContents(False)
        self.label_name = QLabel("상품명", self)
        self.label_progress = QLabel("(0/0)", self)
        self.label_status = QLabel("대기", self)
        self.setMinimumHeight(self.IMAGE_SIZE.height())
        self.setup_ui()
        self.apply_image(None)

    def setup_ui(self) -> None:
        layout_root = QHBoxLayout(self)
        layout_root.setContentsMargins(0, 0, 0, 0)
        layout_root.setSpacing(12)

        layout_root.addWidget(self.label_image)

        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.label_name)
        center_layout.addWidget(self.label_progress)
        layout_root.addLayout(center_layout, 3)

        self.label_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout_root.addWidget(self.label_status, 1)

    def apply_item(
        self,
        index: int,
        name: str,
        quantity: int,
        status_text: str,
        picked: int = 0,
        image_path: Path | None = None,
        **_: Any,
    ) -> None:
        self.apply_image(image_path)
        self.label_image.setToolTip(f"#{index}")
        self.label_name.setText(name)
        self.label_progress.setText(f"({picked}/{quantity})")
        self.label_status.setText(status_text)

    def apply_image(self, path: Path | None) -> None:
        image_path = path if path and path.exists() else self.FALLBACK_IMAGE
        pixmap = QtGui.QPixmap(str(image_path))
        if pixmap.isNull():
            self.label_image.clear()
            return
        scaled = pixmap.scaled(
            self.IMAGE_SIZE,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.label_image.setPixmap(scaled)
