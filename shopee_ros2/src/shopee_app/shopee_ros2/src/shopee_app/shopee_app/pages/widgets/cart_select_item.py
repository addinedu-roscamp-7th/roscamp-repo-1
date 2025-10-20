from pathlib import Path
from typing import Any

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget


class CartSelectItemWidget(QWidget):
    FALLBACK_IMAGE = (
        Path(__file__).resolve().parent.parent / "image" / "product_no_image.png"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_index = QLabel("", self)
        self.label_index.setFixedSize(48, 48)
        self.label_index.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_index.setScaledContents(False)
        self.label_index.setStyleSheet(
            "border-radius: 8px; background-color: #f6f6f6;"
        )
        self.label_name = QLabel("상품명", self)
        self.label_progress = QLabel("(0/0)", self)
        self.label_status = QLabel("대기", self)
        self.setup_ui()
        self.apply_image(None)

    def setup_ui(self) -> None:
        layout_root = QHBoxLayout(self)
        layout_root.setContentsMargins(0, 0, 0, 0)
        layout_root.setSpacing(8)

        layout_root.addWidget(self.label_index, 1)

        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.label_name)
        center_layout.addWidget(self.label_progress)
        layout_root.addLayout(center_layout, 3)

        self.label_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout_root.addWidget(self.label_status, 1)
        self.setMinimumHeight(56)

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
        self.label_index.setToolTip(f"#{index}")
        self.label_name.setText(name)
        self.label_progress.setText(f"({picked}/{quantity})")
        self.label_status.setText(status_text)

    def apply_image(self, path: Path | None) -> None:
        image_path = path if path and path.exists() else self.FALLBACK_IMAGE
        pixmap = QtGui.QPixmap(str(image_path))
        if pixmap.isNull():
            self.label_index.clear()
            return
        size = self.label_index.size() - QtCore.QSize(4, 4)
        if size.width() <= 0 or size.height() <= 0:
            size = QtCore.QSize(40, 40)
        scaled = pixmap.scaled(
            size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.label_index.setPixmap(scaled)
