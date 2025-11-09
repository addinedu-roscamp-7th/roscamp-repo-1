from pathlib import Path
from typing import Any

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QWidget

from shopee_app.ui_gen.cart_select_item import Ui_cart_select_item


class CartSelectItemWidget(QWidget):
    '''로봇 장바구니 선택 진행 상태를 요약해 보여 주는 위젯.'''

    FALLBACK_IMAGE = (
        Path(__file__).resolve().parent.parent / 'image' / 'product_no_image.png'
    )
    IMAGE_SIZE = QtCore.QSize(40, 40)
    STATUS_ICON_SIZE = QtCore.QSize(24, 24)

    def __init__(self, parent=None):
        '''UI 구성 요소를 초기화하고 상태 애니메이션을 준비한다.'''
        super().__init__(parent)
        self.ui = Ui_cart_select_item()
        self.ui.setupUi(self)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.label_image = self.ui.item_image
        self.label_name = self.ui.item_name
        self.label_progress = self.ui.item_count
        self.label_status = self.ui.item_state
        self._configure_widgets()
        self.apply_image(None)
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self._rotate_progress_icon)
        self.progress_angle = 0
        self.progress_base_pixmap: QtGui.QPixmap | None = None
        self.current_status = '대기'

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
        '''주문 항목 정보를 적용하고 진행률 표시를 갱신한다.'''
        self.apply_image(image_path)
        self.label_image.setToolTip(f'#{index}')
        self.label_name.setText(name)
        self.label_name.setToolTip(f'#{index} {name}')
        self.label_progress.setText(f'({picked}/{quantity})')
        self.set_status(status_text)

    def sizeHint(self) -> QtCore.QSize:
        '''레이아웃이 위젯 높이를 과도하게 키우지 않도록 권장 크기를 지정한다.'''
        layout = self.layout()
        spacing = layout.spacing() if layout is not None else 0
        margins = layout.contentsMargins() if layout is not None else QtCore.QMargins()
        text_height = (
            self.label_name.sizeHint().height()
            + self.label_progress.sizeHint().height()
            + spacing
        )
        content_height = text_height
        total_height = content_height + margins.top() + margins.bottom()
        image_width = max(self.IMAGE_SIZE.width(), self.label_image.sizeHint().width())
        status_width = max(self.STATUS_ICON_SIZE.width(), self.label_status.sizeHint().width())
        text_width = max(
            self.label_name.sizeHint().width(),
            self.label_progress.sizeHint().width(),
        )
        spacing = layout.spacing() if layout is not None else 0
        base_width = (
            image_width
            + spacing
            + text_width
            + spacing
            + status_width
            + margins.left()
            + margins.right()
        )
        clamped_width = max(base_width, 160)
        return QtCore.QSize(clamped_width, total_height)

    def _configure_widgets(self) -> None:
        '''위젯 크기 정책과 정렬을 조정해 레이아웃을 안정화한다.'''
        image_policy = self.label_image.sizePolicy()
        image_policy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        image_policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.label_image.setSizePolicy(image_policy)
        if self.label_image.minimumWidth() <= 0 or self.label_image.minimumHeight() <= 0:
            self.label_image.setMinimumSize(self.IMAGE_SIZE)
        if self.label_image.maximumWidth() <= 0 or self.label_image.maximumHeight() <= 0:
            self.label_image.setMaximumSize(self.IMAGE_SIZE)
        if not self.label_image.alignment() & QtCore.Qt.AlignmentFlag.AlignHCenter:
            self.label_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_image.setScaledContents(False)

        if not self.label_status.alignment() & QtCore.Qt.AlignmentFlag.AlignHCenter:
            self.label_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_status.setAccessibleName('선택 상태 표시 아이콘')
        self.label_status.setScaledContents(False)

    def apply_image(self, path: Path | None) -> None:
        '''이미지 경로를 적용하고 없을 경우 기본 이미지를 사용한다.'''
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

    def set_status(self, status_text: str) -> None:
        '''상태에 따라 아이콘 또는 텍스트를 설정한다.'''
        self.current_status = status_text
        normalized = status_text.strip()
        compact = normalized.replace(' ', '')
        self.label_status.setToolTip(normalized)
        if self._is_completed_status(compact):
            self._stop_progress_animation()
            pixmap = self._load_icon_pixmap('checked.svg')
            self._apply_status_pixmap(pixmap)
            return
        if self._is_in_progress_status(compact):
            self._start_progress_animation()
            return
        self._stop_progress_animation()
        self.label_status.clear()
        if self._should_display_text_status(compact):
            self.label_status.setText(normalized)

    def _load_icon_pixmap(self, filename: str) -> QtGui.QPixmap | None:
        '''아이콘 파일을 읽어 적절한 크기의 픽스맵으로 반환한다.'''
        icon_path = Path(__file__).resolve().parent.parent / 'icons' / filename
        if not icon_path.exists():
            return None
        target_size = self.label_status.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self.STATUS_ICON_SIZE
        icon = QtGui.QIcon(str(icon_path))
        pixmap = icon.pixmap(target_size)
        if not pixmap.isNull():
            return pixmap
        return self._render_svg(icon_path, target_size)

    def _apply_status_pixmap(self, pixmap: QtGui.QPixmap | None) -> None:
        '''상태 라벨에 픽스맵을 설정하고 없을 경우 텍스트를 유지한다.'''
        self.label_status.clear()
        if pixmap is None or pixmap.isNull():
            self.label_status.setText(self.current_status)
            return
        scaled = pixmap.scaled(
            self.label_status.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.label_status.setPixmap(scaled)

    def _start_progress_animation(self) -> None:
        '''진행 중 상태에 맞춰 회전 애니메이션을 시작한다.'''
        self.progress_base_pixmap = self._load_icon_pixmap('progress.svg')
        if self.progress_base_pixmap is None:
            self.label_status.setText(self.current_status)
            return
        self.progress_angle = 0
        self.label_status.clear()
        self._rotate_progress_icon()
        if not self.status_timer.isActive():
            self.status_timer.start()

    def _stop_progress_animation(self) -> None:
        '''애니메이션 타이머와 기준 픽스맵을 초기화한다.'''
        if self.status_timer.isActive():
            self.status_timer.stop()
        self.progress_base_pixmap = None
        self.progress_angle = 0

    def _rotate_progress_icon(self) -> None:
        '''타이머 이벤트마다 아이콘을 회전시켜 진행감을 준다.'''
        if self.progress_base_pixmap is None:
            if self.status_timer.isActive():
                self.status_timer.stop()
            return
        target_size = self.label_status.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self.STATUS_ICON_SIZE
        rotated = QtGui.QPixmap(target_size)
        rotated.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(rotated)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(target_size.width() / 2, target_size.height() / 2)
        painter.rotate(self.progress_angle)
        painter.translate(-target_size.width() / 2, -target_size.height() / 2)
        source = self.progress_base_pixmap.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        offset_x = (target_size.width() - source.width()) / 2
        offset_y = (target_size.height() - source.height()) / 2
        painter.drawPixmap(int(offset_x), int(offset_y), source)
        painter.end()
        self.label_status.setPixmap(rotated)
        self.progress_angle = (self.progress_angle + 30) % 360

    def _is_completed_status(self, status: str) -> bool:
        '''완료 상태를 나타내는 키워드를 포함하는지 확인한다.'''
        keywords = ('완료', '성공', '종료')
        return any(keyword in status for keyword in keywords)

    def _is_in_progress_status(self, status: str) -> bool:
        '''진행 중 상태를 나타내는지 판별한다.'''
        if not status:
            return False
        if '대기' in status or '준비' in status:
            return False
        progress_keywords = (
            '진행',
            '선택중',
            '선택중입니다',
            '처리중',
            '집는중',
            '담는중',
            '이동중',
            '작업중',
            '동작중',
        )
        if any(keyword in status for keyword in progress_keywords):
            return True
        if status.endswith('중') and '대기중' not in status:
            return True
        return False

    def _should_display_text_status(self, status: str) -> bool:
        '''텍스트 경고가 필요한 상태인지 판단한다.'''
        if not status:
            return False
        alert_keywords = (
            '실패',
            '오류',
            '중단',
            '에러',
            '경고',
        )
        return any(keyword in status for keyword in alert_keywords)

    def _render_svg(self, path: Path, target_size: QtCore.QSize) -> QtGui.QPixmap | None:
        '''SVG 아이콘을 직접 렌더링해 픽스맵으로 변환한다.'''
        renderer = QSvgRenderer(str(path))
        if not renderer.isValid():
            return None
        pixmap = QtGui.QPixmap(target_size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        return pixmap
