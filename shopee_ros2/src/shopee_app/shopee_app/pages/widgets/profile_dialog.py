from pathlib import Path
from typing import Any

from PyQt6 import QtCore
from PyQt6 import QtWidgets
from PyQt6 import uic


UI_PATH = Path(__file__).resolve().parent.parent.parent / 'ui' / 'dialog_profile.ui'

ALLERGY_LABELS = {
    'nuts': '견과류',
    'milk': '우유',
    'seafood': '해산물',
    'soy': '대두',
    'peach': '복숭아',
    'gluten': '글루텐',
    'eggs': '계란',
}


class ProfileDialog(QtWidgets.QDialog):
    logout_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self.configure_window()
        self._user_info: dict[str, Any] = {}
        logout_button = getattr(self, 'btn_logout', None)
        if logout_button is not None:
            logout_button.clicked.connect(self._on_logout_clicked)

    def configure_window(self) -> None:
        self.setWindowFlags(
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(260, 220)

    def set_user_info(self, user_info: dict[str, Any]) -> None:
        self._user_info = dict(user_info or {})
        name = str(self._user_info.get('name') or self._user_info.get('user_id') or '사용자')
        if hasattr(self, 'label_user_name'):
            self.label_user_name.setText(name)

        gender_label = getattr(self, 'labal_gender', None)
        if gender_label is not None:
            gender_value = self._user_info.get('gender')
            gender_text = '-'
            if gender_value is not None:
                gender_text = '남' if self._is_truthy(gender_value) else '여'
            gender_label.setText(gender_text)

        allergy_label = getattr(self, 'label_allergy_info', None)
        if allergy_label is not None:
            allergy_info = self._user_info.get('allergy_info') or {}
            active = [label for key, label in ALLERGY_LABELS.items() if self._is_truthy(allergy_info.get(key))]
            allergy_label.setText(' / '.join(active) if active else '없음')

        vegan_label = getattr(self, 'label_is_vegan', None)
        if vegan_label is not None:
            vegan_value = self._user_info.get('is_vegan')
            if vegan_value is None:
                vegan_label.setText('-')
            else:
                vegan_label.setText('예' if self._is_truthy(vegan_value) else '아니오')

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        if isinstance(value, str):
            if value.lower() in {'true', '1', 'yes'}:
                return True
            if value.lower() in {'false', '0', 'no'}:
                return False
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, bool):
            return value
        return bool(value)

    def closeEvent(self, event: QtCore.QEvent) -> None:
        super().closeEvent(event)

    def _on_logout_clicked(self) -> None:
        self.logout_requested.emit()
        self.accept()
