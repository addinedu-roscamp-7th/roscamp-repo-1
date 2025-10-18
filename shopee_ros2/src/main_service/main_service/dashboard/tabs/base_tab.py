"""모든 탭 위젯의 기반이 되는 BaseTab 클래스"""
from PyQt6.QtWidgets import QWidget

class BaseTab(QWidget):
    """모든 탭들의 부모 클래스"""
    def update_data(self, data):
        """데이터를 받아 UI를 업데이트하는 메서드. 자식 클래스에서 구현해야 합니다."""
        raise NotImplementedError
