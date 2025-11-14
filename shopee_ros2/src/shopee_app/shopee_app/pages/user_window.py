import math
import os
import random
import sys
from pathlib import Path
from dataclasses import replace
from threading import Event
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

from shopee_app.utils.logging import ComponentLogger

import yaml
from PyQt6 import QtCore
from PyQt6.QtGui import QIcon
from PyQt6.QtGui import QPainter
from PyQt6.QtGui import QPixmap
from PyQt6.QtGui import QTransform
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtGui import QImage
from PyQt6.QtGui import QPen
from PyQt6.QtGui import QBrush
from PyQt6.QtGui import QColor
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView
from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QSpacerItem
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from PyQt6.QtWidgets import QGraphicsScene
from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtWidgets import QGraphicsEllipseItem
from PyQt6.QtWidgets import QGraphicsLineItem
from PyQt6.QtWidgets import QGraphicsPixmapItem
from PyQt6.QtWidgets import QGraphicsSimpleTextItem
from PyQt6.QtWidgets import QGraphicsItem
from PyQt6.QtWidgets import QGraphicsRectItem
from PyQt6.QtWidgets import QGraphicsSceneMouseEvent

from shopee_app.services.app_notification_client import AppNotificationClient
from shopee_app.services.main_service_client import MainServiceClient
from shopee_app.services.main_service_client import MainServiceClientError
from shopee_app.services.video_stream_client import VideoStreamReceiver
from shopee_app.pages.models.cart_item_data import CartItemData
from shopee_app.pages.models.product_data import ProductData
from shopee_app.pages.models.allergy_info_data import AllergyInfoData
from shopee_app.pages.widgets.cart_item import CartItemWidget
from shopee_app.pages.widgets.cart_select_item import CartSelectItemWidget
from shopee_app.pages.widgets.product_card import ProductCard
from shopee_app.ui_gen.layout_user import Ui_Form_user as Ui_UserLayout
from shopee_app.pages.widgets.profile_dialog import ProfileDialog
from shopee_app.styles.constants import COLORS
from shopee_app.styles.constants import STYLES

if TYPE_CHECKING:
    from shopee_app.ros_node import RosNodeThread
    from shopee_interfaces.msg import PickeeRobotStatus

_LLM_SERVICE_DIR = Path(__file__).resolve().parents[5] / "shopee_llm" / "LLM_Service"
if _LLM_SERVICE_DIR.is_dir() and str(_LLM_SERVICE_DIR) not in sys.path:
    sys.path.append(str(_LLM_SERVICE_DIR))

from STT_module import STT_Module


DEFAULT_MAP_CONFIG = (
    Path(__file__).resolve().parents[5]
    / "shopee_ros2"
    / "src"
    / "pickee_mobile"
    / "map"
    / "map1021_modify.yaml"
)
DEFAULT_IMAGE_FALLBACK = Path(__file__).resolve().parent / "image" / "map.png"
VECTOR_ICON_PATH = Path(__file__).resolve().parent / "icons" / "vector.svg"
DEFAULT_ROBOT_ICON_SIZE = (24, 24)
SECTION_FRIENDLY_NAMES = {
    "SECTION_1": "기성품",
    "SECTION_7": "신선식품",
    "SECTION_11": "과자",
    "SECTION_C_3": "이클립스",
    "SECTION_B_3": "생선",
}


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_float_optional(name: str) -> float | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _get_env_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parts = [float(part.strip()) for part in value.split(",")]
        return tuple(parts)
    except ValueError:
        return default


MAP_CONFIG_PATH = Path(os.getenv("SHOPEE_APP_MAP_CONFIG", str(DEFAULT_MAP_CONFIG)))
MAP_IMAGE_PATH = os.getenv("SHOPEE_APP_MAP_IMAGE")
MAP_RESOLUTION_OVERRIDE = _get_env_float("SHOPEE_APP_MAP_RESOLUTION", -1.0)
MAP_ORIGIN_OVERRIDE = _get_env_tuple("SHOPEE_APP_MAP_ORIGIN", ())
VIDEO_STREAM_DEBUG = os.getenv("SHOPEE_APP_DEBUG_VIDEO_STREAM", "0") == "1"
ROBOT_OFFSET_X_OVERRIDE = _get_env_float_optional("SHOPEE_APP_ROBOT_OFFSET_X")
ROBOT_OFFSET_Y_OVERRIDE = _get_env_float_optional("SHOPEE_APP_ROBOT_OFFSET_Y")
ROBOT_LABEL_OFFSET_OVERRIDE = _get_env_float_optional("SHOPEE_APP_ROBOT_LABEL_OFFSET_Y")
ROBOT_SCALE_OVERRIDE = _get_env_float_optional("SHOPEE_APP_ROBOT_POSITION_SCALE")


class ClickableLabel(QLabel):
    """마우스 클릭 이벤트를 시그널로 제공하는 QLabel 확장."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """왼쪽 버튼이 떼어질 때 클릭 신호를 방출한다."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class BBoxGraphicsRectItem(QGraphicsRectItem):
    """bbox 사각형을 클릭했을 때 콜백을 호출하는 그래픽 항목."""

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        bbox_number: int,
        on_click: Callable[[int], None] | None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(x, y, width, height, parent)
        self._bbox_number = bbox_number
        self._on_click = on_click
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """좌클릭 시 선택 콜백을 호출한다."""
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._bbox_number > 0
            and self._on_click is not None
        ):
            self._on_click(self._bbox_number)
            event.accept()
            return
        super().mousePressEvent(event)


class SttStatusDialog(QDialog):
    """음성 인식 진행 상황을 실시간으로 보여 주는 모달 대화상자."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("음성 인식")
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowType.Tool)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        self.message_label = QLabel("", self)
        layout.addWidget(self.message_label)

    def update_message(self, text: str, *, warning: bool = False) -> None:
        """상태 메시지를 갱신하고 경고 여부에 따라 색상을 조정한다."""
        # 안내 문구를 업데이트하면서 경고 여부에 따라 색상을 바꾼다.
        self.message_label.setText(text)
        if warning:
            self.message_label.setStyleSheet("color: #d32f2f;")
        else:
            self.message_label.setStyleSheet("")
        self.adjustSize()
        self.repaint()


class SttWorker(QtCore.QObject):
    """Whisper STT를 별도 스레드에서 실행해 UI 응답성을 유지한다."""

    microphone_detected = QtCore.pyqtSignal(int, str)
    listening_started = QtCore.pyqtSignal()
    result_ready = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        stt_module: STT_Module,
        detect_microphone: Callable[[STT_Module], tuple[int, str] | None],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._stt_module = stt_module
        self._detect_microphone = detect_microphone

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """마이크를 탐색하고 STT를 실행한 뒤 결과를 발행한다."""
        try:
            # 우선 사용 가능한 마이크를 탐색하고 발견한 정보를 신호로 알린다.
            microphone_info = self._detect_microphone(self._stt_module)
            if microphone_info is None:
                self.error_occurred.emit("사용 가능한 마이크 정보를 찾지 못했습니다.")
                return
            microphone_index, microphone_name = microphone_info
            self.microphone_detected.emit(microphone_index, microphone_name)
            prompt_notified = False

            def _notify_prompt() -> None:
                # Whisper가 프롬프트 문구를 출력하기 전까지 단 한 번만 듣기 시작 신호를 보낸다.
                nonlocal prompt_notified
                if prompt_notified:
                    return
                prompt_notified = True
                self.listening_started.emit()

            original_stdout = sys.stdout

            class _StdoutProxy:
                def __init__(self, target: Any, callback: Callable[[], None]) -> None:
                    self._target = target
                    self._callback = callback

                def write(self, text: str) -> int:
                    # Whisper 모듈이 stdout에 프롬프트를 쓰는 순간 콜백을 호출해 UI에 반영한다.
                    self._target.write(text)
                    self._target.flush()
                    if "Please Talk to me" in text:
                        self._callback()
                    return len(text)

                def flush(self) -> None:
                    self._target.flush()

            sys.stdout = _StdoutProxy(original_stdout, _notify_prompt)
            try:
                # Whisper STT 실행 결과를 문자열로 변환해 신호로 전달한다.
                result = self._stt_module.stt_use()
            finally:
                sys.stdout = original_stdout
            if not prompt_notified:
                self.listening_started.emit()
            if not isinstance(result, str):
                result = "" if result is None else str(result)
            self.result_ready.emit(result)
        except SystemExit:
            self.error_occurred.emit("음성 인식이 중단되었습니다.")
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()


class UserWindow(QWidget):
    """사용자 쇼핑 화면과 로봇 제어 흐름을 담당하는 주 창."""

    IMAGE_ROOT = Path(__file__).resolve().parent / "image"
    ROBOT_ICON_ROTATION_OFFSET_DEG = 90.0
    MAP_TOP_LEFT = (6.3, 8.4)
    MAP_BOTTOM_RIGHT = (-5.8, 0.5)
    ROBOT_POSITION_OFFSET_X = (
        ROBOT_OFFSET_X_OVERRIDE if ROBOT_OFFSET_X_OVERRIDE is not None else -0.1
    )
    ROBOT_POSITION_OFFSET_Y = (
        ROBOT_OFFSET_Y_OVERRIDE if ROBOT_OFFSET_Y_OVERRIDE is not None else 1.5
    )
    ROBOT_POSITION_SCALE = (
        ROBOT_SCALE_OVERRIDE if ROBOT_SCALE_OVERRIDE is not None else 1.0
    )
    SHOW_ROBOT_LABEL = False
    ROBOT_LABEL_OFFSET_Y = (
        ROBOT_LABEL_OFFSET_OVERRIDE
        if ROBOT_LABEL_OFFSET_OVERRIDE is not None
        else -36.0
    )
    VIDEO_FRAME_WIDTH = 640
    VIDEO_FRAME_HEIGHT = 480
    PRODUCT_IMAGE_BY_ID: dict[int, Path] = {
        1: IMAGE_ROOT / "product_horseradish.png",
        2: IMAGE_ROOT / "product_spicy_chicken.png",
        4: IMAGE_ROOT / "product_richam.png",
        5: IMAGE_ROOT / "product_soymilk.png",
        6: IMAGE_ROOT / "product_caprisun.png",
        7: IMAGE_ROOT / "product_apple.png",
        8: IMAGE_ROOT / "product_green_apple.png",
        9: IMAGE_ROOT / "product_ivy.png",
        10: IMAGE_ROOT / "product_pork.png",
        11: IMAGE_ROOT / "product_chicken.png",
        12: IMAGE_ROOT / "product_mackerel.png",
        13: IMAGE_ROOT / "product_abalone.png",
        14: IMAGE_ROOT / "product_eclips.png",
        16: IMAGE_ROOT / "product_pepero.png",
        17: IMAGE_ROOT / "product_oyes.png",
        18: IMAGE_ROOT / "product_orange.png",
        19: IMAGE_ROOT / "product_jangjorim.png",
    }
    PRODUCT_IMAGE_BY_NAME: dict[str, Path] = {
        "고추냉이": IMAGE_ROOT / "product_horseradish.png",
        "버터캔": IMAGE_ROOT / "product_jangjorim.png",
        "리챔": IMAGE_ROOT / "product_richam.png",
        "두유": IMAGE_ROOT / "product_soymilk.png",
        "카프리썬": IMAGE_ROOT / "product_caprisun.png",
        "홍사과": IMAGE_ROOT / "product_apple.png",
        "청사과": IMAGE_ROOT / "product_green_apple.png",
        "삼겹살": IMAGE_ROOT / "product_pork.png",
        "닭": IMAGE_ROOT / "product_chicken.png",
        "생선": IMAGE_ROOT / "product_mackerel.png",
        "전복": IMAGE_ROOT / "product_abalone.png",
        "이클립스": IMAGE_ROOT / "product_eclips.png",
        "빼빼로": IMAGE_ROOT / "product_pepero.png",
        "오예스": IMAGE_ROOT / "product_oyes.png",
        "아이비": IMAGE_ROOT / "product_ivy.png",
        "오렌지": IMAGE_ROOT / "product_orange.png",
        "불닭캔": IMAGE_ROOT / "product_spicy_chicken.png",
    }
    closed = pyqtSignal()

    def __init__(
        self,
        *,
        user_info: dict[str, Any] | None = None,
        service_client: MainServiceClient | None = None,
        ros_thread: "RosNodeThread | None" = None,
        parent=None,
    ):
        """사용자 정보와 서비스 의존성을 받아 UI를 초기화한다."""
        super().__init__(parent)

        # 컴포넌트 로거 초기화
        self.logger = ComponentLogger("user_window")
        self._footer_locked = False

        # UI 초기화
        self.ui = Ui_UserLayout()
        self.ui.setupUi(self)

        # 정리 플래그
        self._cleanup_requested = Event()

        self.setWindowTitle("Shopee GUI - User")
        self.products_container = getattr(self.ui, "grid_products", None)
        self.product_grid = getattr(self.ui, "gridLayout_2", None)
        self.products: list[ProductData] = []
        self.all_products: list[ProductData] = []
        self.product_index: dict[int, ProductData] = {}
        self.default_empty_products_message = "표시할 상품이 없습니다."
        self.empty_products_message = self.default_empty_products_message
        self.filtered_empty_message = "선택한 필터 조건에 맞는 상품이 없습니다."

        self.cart_items: dict[int, CartItemData] = {}
        self.cart_widgets: dict[int, CartItemWidget] = {}
        self.cart_items_layout = getattr(self.ui, "cart_items_layout", None)
        self.cart_spacer = None
        self.cart_empty_label: QLabel | None = None
        if self.cart_items_layout is not None:
            # 장바구니가 비었을 때 표시할 안내 문구를 준비한다.
            self.cart_empty_label = QLabel("장바구니에 담긴 상품이 없습니다.")
            self.cart_empty_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignHCenter
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.cart_empty_label.setStyleSheet(
                "color: #9AA0A6; font-size: 12pt; font-weight: 500;"
            )
            self.cart_empty_label.setWordWrap(True)
            self.cart_empty_label.hide()
            self.cart_items_layout.insertWidget(0, self.cart_empty_label)
            self.cart_spacer = QSpacerItem(
                0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
            )
            self.cart_items_layout.addItem(self.cart_spacer)
        self.ui.chk_select_all.stateChanged.connect(self.on_select_all_changed)
        if hasattr(self.ui, "btn_selected_delete"):
            self.ui.btn_selected_delete.clicked.connect(self.on_delete_selected_clicked)

        # 맵 좌표 설정
        self.WORLD_X_MIN = 0.5
        self.WORLD_X_MAX = 8.4
        self.WORLD_Y_MAX = 6.3  # 위
        self.WORLD_Y_MIN = -5.8  # 아래

        self.current_columns = -1
        self.cart_container = None
        self.cart_frame = None
        self.cart_body = None
        self.cart_header = None
        self.cart_toggle_button = None
        self.cart_icon_up: QIcon | None = None
        self.cart_icon_down: QIcon | None = None
        self.product_scroll = None
        self.cart_expanded = False
        self.cart_container_layout = None
        self.cart_margin_left = 0
        self.cart_margin_right = 0
        self.cart_margin_bottom = 0
        self.cart_margin_collapsed_top = 0
        self.cart_margin_expanded_top = 24
        self.main_stack = None
        self.side_stack = None
        self.page_user = None
        self.page_pick = None
        self.side_shop_page = None
        self.side_pick_filter_page = None
        self.shopping_button = None
        self.store_button = None
        self.nav_group = None
        self.current_mode = None
        self.selection_container = getattr(self.ui, "widget_selection_container", None)
        self.selection_grid = getattr(self.ui, "grid_selection_buttons_2", None)
        self.refresh_logo_label: ClickableLabel | None = None
        self.selection_buttons: list[QPushButton] = []
        self.selection_button_group = QButtonGroup(self)
        self.selection_button_group.setExclusive(True)
        self.selection_button_group.buttonToggled.connect(
            self.on_selection_button_toggled
        )
        self.selection_options: list[dict[str, Any]] = []
        self.selection_selected_index: int | None = None
        self.selection_options_origin: str | None = None
        self.ros_thread = ros_thread
        self.pose_tracking_active = False
        self._set_detection_subscription(False)
        self.order_select_stack = getattr(self.ui, "stackedWidget", None)
        self.page_select_product = getattr(self.ui, "page_select_product", None)
        self.page_moving_view = getattr(self.ui, "page_moving", None)
        self.page_end_pick_up = getattr(self.ui, "page_end_pick_up", None)
        self.pick_flow_completed = False
        self.progress_bar = getattr(self.ui, "shop_progressBar", None)
        self.progress_text = getattr(self.ui, "text_progress", None)
        self.front_view_button: QPushButton | None = None
        self.arm_view_button: QPushButton | None = None
        self.video_stack = getattr(self.ui, "stackedWidget_3", None)
        self.page_front = getattr(self.ui, "page_front", None)
        self.page_arm = getattr(self.ui, "page_arm", None)
        self.gv_front = getattr(self.ui, "gv_front", None)
        self.gv_arm = getattr(self.ui, "gv_arm", None)
        self.map_view = getattr(self.ui, "gv_map", None)
        self.front_scene: QGraphicsScene | None = None
        self.arm_scene: QGraphicsScene | None = None
        self.front_item = None
        self.arm_item = None
        self.bbox_overlay_items: list[QGraphicsItem] = []
        self.bbox_rect_items: dict[int, QGraphicsRectItem] = {}
        self.bbox_label_items: dict[int, QGraphicsSimpleTextItem] = {}
        self.bbox_label_bg_items: dict[int, QGraphicsRectItem] = {}
        self.pending_detection_products: list[dict[str, Any]] = []
        self._bbox_overlays_dirty = False
        self.map_scene: QGraphicsScene | None = None
        self.map_pixmap_item: QGraphicsPixmapItem | None = None
        self.map_robot_item: QGraphicsPixmapItem | QGraphicsEllipseItem | None = None
        self.map_heading_item: QGraphicsLineItem | None = None
        self.map_robot_label: QGraphicsSimpleTextItem | None = None
        self.map_resolution: float | None = None
        self.map_origin: tuple[float, float] | None = None
        self.map_image_size: tuple[int, int] | None = None
        self.video_receiver: VideoStreamReceiver | None = None
        self.active_camera_type: str | None = None
        self._auto_camera_warning_displayed = False
        self.allergy_toggle_button: QPushButton | None = None
        self.allergy_sub_widget: QWidget | None = None
        self.allergy_filters_expanded = True
        self._allergy_max_height = 16777215
        self.allergy_icon_fold: QIcon | None = None
        self.allergy_icon_fold_rotated: QIcon | None = None
        self.allergy_total_checkbox: QCheckBox | None = None
        self.allergy_checkbox_map: dict[str, QCheckBox] = {}
        self.vegan_checkbox: QCheckBox | None = None
        self._init_video_views()
        self._init_map_view()
        self.select_title_label = getattr(self.ui, "label_7", None)
        self.select_done_button = getattr(self.ui, "btn_add_product", None)
        if self.select_done_button is None:
            self.select_done_button = getattr(self.ui, "toolButton", None)
        if self.select_done_button is not None:
            self.select_done_button.clicked.connect(self.on_select_done_clicked)
            self.select_done_button.setStyleSheet(
                f"""
                    background-color: {COLORS['primary']};
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 24px;
                """
            )
        self.selection_state_label = getattr(self.ui, "label_selecting_state", None)
        if self.selection_state_label is not None:
            self.selection_state_label.setText("선택 대기 중")
        self.selection_voice_button = getattr(self.ui, "btn_select_cancel", None)
        if self.selection_voice_button is None:
            self.selection_voice_button = getattr(self.ui, "toolButton_4", None)
        if self.selection_voice_button is not None:
            self.selection_voice_button.setCheckable(True)
            self.selection_voice_button.setChecked(False)
            self.selection_voice_button.setToolTip("음성으로 선택지 고르기")
            mic_icon_path = Path(__file__).resolve().parent / "icons" / "mic.svg"
            if mic_icon_path.exists():
                self.selection_voice_button.setIcon(QIcon(str(mic_icon_path)))
                self.selection_voice_button.setIconSize(QtCore.QSize(20, 20))
            if hasattr(self.selection_voice_button, "setToolButtonStyle"):
                self.selection_voice_button.setToolButtonStyle(
                    QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
                )
            if hasattr(self.selection_voice_button, "setSizePolicy"):
                self.selection_voice_button.setSizePolicy(
                    QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
                )
            primary_color = COLORS["primary"]
            neutral_color = "#666666"
            self.selection_voice_button.setStyleSheet(
                f"QToolButton {{ text-align: center; padding: 6px 16px; min-height: 40px; border-radius: 10px; "
                f"background-color: transparent; color: {neutral_color}; border: 1px solid {neutral_color}; "
                f"margin-right: 12px; }} "
                f"QToolButton:checked {{ background-color: {primary_color}; color: white; border-color: {primary_color}; }} "
                f"QToolButton::icon {{ margin-right: 3px; }}"
            )
            self._update_selection_voice_button(False)
            self.selection_voice_button.clicked.connect(
                self.on_selection_voice_button_clicked
            )
        self.auto_pick_button: QPushButton | None = getattr(
            self.ui, "btn_auto_pick", None
        )
        if self.auto_pick_button is not None:
            self.auto_pick_button.setEnabled(False)
            self.auto_pick_button.clicked.connect(self.on_auto_pick_clicked)
        self.front_view_button = getattr(self.ui, "btn_view_front", None)
        self.arm_view_button = getattr(self.ui, "btn_view_arm", None)
        self.camera_button_group = QButtonGroup(self)
        self.camera_button_group.setExclusive(True)
        if self.front_view_button is not None:
            self.front_view_button.setCheckable(True)
            self.camera_button_group.addButton(self.front_view_button)
            self.front_view_button.clicked.connect(
                lambda: self.on_camera_button_clicked("front")
            )
        if self.arm_view_button is not None:
            self.arm_view_button.setCheckable(True)
            self.camera_button_group.addButton(self.arm_view_button)
            self.arm_view_button.clicked.connect(
                lambda: self.on_camera_button_clicked("arm")
            )
        self.pick_bottom_sheet_frame = getattr(self.ui, "frame_pick_bottom_sheet", None)
        if self.pick_bottom_sheet_frame is not None:
            self._apply_pick_bottom_sheet_frame_style(self.pick_bottom_sheet_frame)
        self.shop_end_button = getattr(self.ui, "btn_shop_end", None)
        if self.shop_end_button is not None:
            self.shop_end_button.clicked.connect(self.on_shop_end_clicked)
            self._apply_shop_end_button_style(self.shop_end_button)
        self.shop_continue_button = getattr(self.ui, "btn_shop_continue", None)
        if self.shop_continue_button is not None:
            self.shop_continue_button.clicked.connect(self.on_shop_continue_clicked)
            self._apply_shop_continue_button_style(self.shop_continue_button)
        # 검색 입력 위젯을 저장하지 않으면 사용자가 입력한 검색어를 가져올 방법이 없다.
        self.search_input = getattr(self.ui, "edit_search", None)
        if self.search_input is None:
            self.search_input = getattr(self.ui, "lineEdit", None)
        # 위젯 존재 여부를 검증하지 않으면 None 객체에 연결을 시도해 런타임 오류가 발생한다.
        if self.search_input is not None:
            # 엔터 키 입력 시 검색을 자동으로 수행하지 않으면 사용자의 검색 흐름이 끊어진다.
            self.search_input.returnPressed.connect(self.on_search_submitted)
        self.search_button = getattr(self.ui, "btn_search", None)
        if self.search_button is not None:
            # 버튼을 눌렀을 때 검색이 실행되지 않으면 사용자가 직관적으로 조작하기 어렵다.
            self.search_button.clicked.connect(self.on_search_button_clicked)
        self.mic_button = getattr(self.ui, "btn_mic", None)
        if self.mic_button is not None:
            mic_icon_path = Path(__file__).resolve().parent / "icons" / "mic.svg"
            if mic_icon_path.exists():
                self.mic_button.setIcon(QIcon(str(mic_icon_path)))
                self.mic_button.setIconSize(QtCore.QSize(24, 24))
            self.mic_button.setText("")
            self.mic_button.setToolTip("음성으로 검색")
            self.mic_button.clicked.connect(self.on_microphone_clicked)
        self.mic_info_label = getattr(self.ui, "label_mic_info", None)
        if self.mic_info_label is not None:
            self.mic_info_label.setText("")
            self.mic_info_label.setVisible(False)

        self._setup_refresh_logo()
        self.setup_cart_section()
        self.setup_navigation()
        self._setup_allergy_toggle()
        self._setup_allergy_checkboxes()
        self._apply_main_banner_image()
        self._style_main_banner_title()
        self._style_do_create_label()
        if self.ros_thread is not None:
            self.ros_thread.pickee_status_received.connect(
                self._on_pickee_status_received
            )
            try:
                self.ros_thread.pickee_detection_received.connect(
                    self._on_detection_received
                )
            except AttributeError:
                pass
        from shopee_app.styles.constants import STYLES

        self.ui.btn_pay.setStyleSheet(STYLES["pay_button"])
        self.ui.btn_pay.clicked.connect(self.on_pay_clicked)
        self._stt_feedback_timer = QtCore.QTimer(self)
        self._stt_feedback_timer.setInterval(1000)
        self._stt_feedback_timer.timeout.connect(self._on_stt_feedback_tick)
        self._stt_status_hide_timer = QtCore.QTimer(self)
        self._stt_status_hide_timer.setSingleShot(True)
        self._stt_status_hide_timer.timeout.connect(self._hide_mic_info)
        self._stt_module: STT_Module | None = None
        self._stt_busy = False
        self._stt_status_dialog: SttStatusDialog | None = None
        self._stt_thread: QtCore.QThread | None = None
        self._stt_worker: SttWorker | None = None
        self._stt_status_close_timer = QtCore.QTimer(self)
        self._stt_status_close_timer.setSingleShot(True)
        self._stt_status_close_timer.timeout.connect(self._close_stt_status_dialog)
        self._stt_last_microphone_name: str | None = None
        self._stt_context: str | None = None
        self.empty_products_message = self.default_empty_products_message
        self.set_products([])
        self.update_cart_summary()

        self.user_info: dict[str, Any] = dict(user_info or {})
        self.service_client = (
            service_client if service_client is not None else MainServiceClient()
        )
        self.current_user_id = ""
        self._ensure_user_identity()
        self.current_order_id: int | None = None
        self.current_robot_id: int | None = None
        self.remote_selection_items: list[CartItemData] = []
        self.auto_selection_items: list[CartItemData] = []
        self.selection_item_states: dict[int, dict[str, object]] = {}

        self.profile_dialog = ProfileDialog(self)
        self.profile_dialog.set_user_info(self.user_info)
        self.profile_dialog.logout_requested.connect(self.on_logout_requested)
        profile_button = getattr(self.ui, "btn_profile", None)
        if profile_button is not None:
            icon_path = Path(__file__).resolve().parent / "icons" / "user.svg"
            if icon_path.exists():
                profile_button.setIcon(QIcon(str(icon_path)))
                profile_button.setIconSize(QtCore.QSize(32, 32))
            profile_button.setText("")
            if hasattr(profile_button, "setFlat"):
                profile_button.setFlat(True)
            elif hasattr(profile_button, "setAutoRaise"):
                profile_button.setAutoRaise(True)
            profile_button.setToolTip("프로필 보기")
            profile_button.clicked.connect(self.open_profile_dialog)
        QtCore.QTimer.singleShot(0, self.refresh_product_grid)
        # 초기 화면에서 바로 서버 상품을 갱신하지 않으면 빈 상태가 유지된다.
        QtCore.QTimer.singleShot(0, self.request_total_products)

        self._update_user_header()
        self.notification_client: AppNotificationClient | None = None
        self._initialize_selection_grid()

    def _ensure_user_identity(self) -> str:
        # 로그인 여부와 관계없이 상품 검색을 테스트할 수 있도록 게스트 ID를 제공한다.
        user_id_value = ""
        if isinstance(self.user_info, dict):
            raw_id = self.user_info.get("user_id")
            if raw_id:
                user_id_value = str(raw_id).strip()
        if not user_id_value:
            fallback_user_id = os.getenv("SHOPEE_APP_GUEST_USER_ID", "guest_user")
            user_id_value = fallback_user_id
            if isinstance(self.user_info, dict):
                self.user_info["user_id"] = user_id_value
        self.current_user_id = user_id_value
        return user_id_value

    def _get_stt_module(self) -> STT_Module:
        # STT 모듈은 초기화 비용이 크므로 한 번만 생성해 재사용한다.
        if self._stt_module is None:
            self._stt_module = STT_Module()
        return self._stt_module

    def _detect_microphone(self, stt_module: STT_Module) -> tuple[int, str] | None:
        # 마이크 목록을 직접 확인해 사용자에게 안내할 정보를 구성한다.
        mike_util = getattr(stt_module, "mike_handle", None)
        if mike_util is None:
            return None
        mike_list = mike_util.mike_list_return()
        if not mike_list:
            return None
        target_index = mike_util.pick_mike_index(mike_list)
        if target_index is None:
            return None
        target_name = ""
        for index, name in mike_list:
            if index == target_index:
                target_name = name
                break
        return target_index, target_name

    def _show_stt_status_dialog(
        self, text: str, *, icon: QMessageBox.Icon = QMessageBox.Icon.Information
    ) -> None:
        if self._stt_status_dialog is None:
            dialog = SttStatusDialog(self)
            self._stt_status_dialog = dialog
        self._stt_status_close_timer.stop()
        dialog = self._stt_status_dialog
        prefix = "[안내] " if icon != QMessageBox.Icon.Warning else "[경고] "
        dialog.update_message(
            f"{prefix}{text}",
            warning=icon == QMessageBox.Icon.Warning,
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _schedule_stt_status_close(self, timeout_ms: int) -> None:
        self._stt_status_close_timer.stop()
        if timeout_ms <= 0:
            self._close_stt_status_dialog()
            return
        self._stt_status_close_timer.start(timeout_ms)

    def _close_stt_status_dialog(self) -> None:
        self._stt_status_close_timer.stop()
        if self._stt_status_dialog is None:
            return
        dialog = self._stt_status_dialog
        self._stt_status_dialog = None
        dialog.close()
        dialog.deleteLater()

    def _shutdown_speech_recognition(self) -> None:
        self._stt_module = None
        self._stt_busy = False
        self._stt_feedback_timer.stop()
        self._stt_status_hide_timer.stop()
        if self._stt_thread is not None:
            if self._stt_thread.isRunning():
                self._stt_thread.requestInterruption()
                self._stt_thread.quit()
                self._stt_thread.wait()
            self._stt_thread = None
        self._stt_worker = None
        self._stt_last_microphone_name = None
        self._close_stt_status_dialog()
        if self.mic_button is not None:
            self.mic_button.setEnabled(True)
        if self.selection_voice_button is not None:
            self.selection_voice_button.setEnabled(True)
        self._stt_context = None
        self._update_selection_voice_button(False)
        self.unsetCursor()
        self._hide_mic_info()

    def closeEvent(self, event):
        if self.notification_client is not None:
            self.notification_client.stop()
        self._shutdown_speech_recognition()
        self._stop_video_stream(send_request=True)
        if self.ros_thread is not None:
            try:
                self.ros_thread.pickee_status_received.disconnect(
                    self._on_pickee_status_received
                )
            except TypeError:
                pass
            try:
                self.ros_thread.pickee_detection_received.disconnect(
                    self._on_detection_received
                )
            except TypeError:
                pass
        self._disable_pose_tracking()
        self.closed.emit()
        super().closeEvent(event)

    def on_pay_clicked(self):
        # 주문 생성 전에 알림 채널을 기동해 초기 이동 이벤트가 누락되지 않도록 한다.
        self._ensure_notification_listener()
        if self.request_create_order():
            self.set_mode("pick")
            # 결제 후 매장 화면으로 전환되면 로봇 위치 추적을 시작한다.
            self._enable_pose_tracking()

    def on_search_submitted(self) -> None:
        # 검색 위젯이 준비되지 않았다면 검색어를 읽어올 수 없어 조용히 종료한다.
        if self.search_input is None:
            return
        # 입력값에서 공백을 제거하지 않으면 의미 없는 공백 검색으로 서버를 불필요하게 호출한다.
        query = self.search_input.text().strip()
        # 검색어로 서버 조회를 하지 않으면 사용자가 요청한 상품 목록을 받아올 수 없다.
        self.request_product_search(query)

    def on_search_button_clicked(self) -> None:
        # 버튼 클릭과 엔터 입력이 동일하게 동작하도록 검색 제출 함수를 재사용한다.
        self.on_search_submitted()

    # 검색 영역에서 마이크 버튼을 누르면 STT 쓰레드를 기동한다.
    def on_microphone_clicked(self) -> None:
        if self._stt_busy:
            QMessageBox.information(
                self, "음성 인식 진행 중", "이미 음성을 인식하고 있습니다."
            )
            return
        self._stt_context = "search"
        self._start_mic_feedback()
        self.on_stt_started()
        self._show_stt_status_dialog(
            "마이크 찾는 중...", icon=QMessageBox.Icon.Information
        )
        stt_module = self._get_stt_module()
        self._stt_last_microphone_name = None
        worker = SttWorker(
            stt_module=stt_module,
            detect_microphone=self._detect_microphone,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.microphone_detected.connect(self._on_stt_microphone_detected)
        worker.listening_started.connect(self._on_stt_listening_started)
        worker.result_ready.connect(self.on_stt_result_ready)
        worker.error_occurred.connect(self.on_stt_error)
        worker.finished.connect(self._on_stt_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_stt_thread_finished)
        self._stt_thread = thread
        self._stt_worker = worker
        thread.start()

    # 상품 선택 화면에서 음성 버튼을 누르면 동일한 STT 파이프라인을 재사용한다.
    def on_selection_voice_button_clicked(self) -> None:
        # 선택지가 없다면 음성 입력으로도 전달할 데이터가 없다.
        if not self.selection_options:
            QMessageBox.information(
                self, "음성 선택", "현재 고를 수 있는 선택지가 없습니다."
            )
            self._update_selection_voice_button(False)
            return
        if self._stt_busy:
            QMessageBox.information(
                self, "음성 선택", "다른 음성 명령이 진행 중입니다."
            )
            self._update_selection_voice_button(False)
            return
        self._stt_context = "selection"
        self._start_mic_feedback()
        self.on_stt_started()
        self._show_stt_status_dialog(
            "마이크 찾는 중...", icon=QMessageBox.Icon.Information
        )
        stt_module = self._get_stt_module()
        self._stt_last_microphone_name = None
        worker = SttWorker(
            stt_module=stt_module,
            detect_microphone=self._detect_microphone,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.microphone_detected.connect(self._on_stt_microphone_detected)
        worker.listening_started.connect(self._on_stt_listening_started)
        worker.result_ready.connect(self.on_stt_result_ready)
        worker.error_occurred.connect(self.on_stt_error)
        worker.finished.connect(self._on_stt_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_stt_thread_finished)
        self._stt_thread = thread
        self._stt_worker = worker
        thread.start()

    @QtCore.pyqtSlot(int, str)
    def _on_stt_microphone_detected(
        self, microphone_index: int, microphone_name: str
    ) -> None:
        self._stt_last_microphone_name = microphone_name
        self._show_stt_status_dialog(
            f"인식된 마이크: [{microphone_index}] {microphone_name}\n 마이크를 초기화하는 중입니다...",
            icon=QMessageBox.Icon.Information,
        )

    @QtCore.pyqtSlot()
    def _on_stt_listening_started(self) -> None:
        self._stt_status_close_timer.stop()
        if self._stt_last_microphone_name:
            prompt = f"마이크: [{self._stt_last_microphone_name}] \n 음성 인식 중..."
        elif self._stt_context == "selection":
            prompt = "선택할 번호를 말씀해주세요..."
        else:
            prompt = "검색할 음성을 말해주세요..."
        self._show_stt_status_dialog(prompt, icon=QMessageBox.Icon.Information)

    @QtCore.pyqtSlot()
    def _on_stt_worker_finished(self) -> None:
        self._stt_worker = None
        self._stt_last_microphone_name = None
        self.on_stt_finished()

    @QtCore.pyqtSlot()
    def _on_stt_thread_finished(self) -> None:
        self._stt_thread = None

    # STT가 시작되면 UI 상호작용을 잠시 비활성화한다.
    def on_stt_started(self) -> None:
        self._stt_busy = True
        if self.mic_button is not None:
            self.mic_button.setEnabled(False)
        if self.selection_voice_button is not None:
            self.selection_voice_button.setEnabled(False)
        if self._stt_context == "selection":
            self._update_selection_voice_button(True)
        self.setCursor(QtCore.Qt.CursorShape.WaitCursor)

    # STT가 종료되면 버튼과 커서를 원래 상태로 되돌린다.
    def on_stt_finished(self) -> None:
        self._stt_busy = False
        current_context = self._stt_context
        self._stt_context = None
        self._stt_feedback_timer.stop()
        if self.mic_button is not None:
            self.mic_button.setEnabled(True)
        if self.selection_voice_button is not None:
            self.selection_voice_button.setEnabled(True)
        if current_context == "selection":
            self._update_selection_voice_button(False)
        self.unsetCursor()
        if (
            self._stt_status_dialog is not None
            and not self._stt_status_close_timer.isActive()
        ):
            self._schedule_stt_status_close(2000)
        if not self._stt_status_hide_timer.isActive():
            self._hide_mic_info()

    def on_stt_result_ready(self, text: str) -> None:
        recognized = text.strip()
        if not recognized:
            self._show_mic_info("음성을 인식하지 못했습니다.")
            self._stt_status_hide_timer.start(2000)
            self._show_stt_status_dialog(
                "음성을 인식하지 못했습니다.",
                icon=QMessageBox.Icon.Information,
            )
            self._schedule_stt_status_close(2000)
            return
        current_context = self._stt_context
        if current_context == "selection":
            handled = self._handle_selection_voice_result(recognized)
            if not handled:
                self._stt_status_hide_timer.start(2500)
            return
        if self.search_input is not None:
            self.search_input.setText(recognized)
        self._show_mic_info(f"음성 인식 결과: {recognized}")
        self._stt_status_hide_timer.start(2500)
        self.request_product_search(recognized)

    def on_stt_error(self, message: str) -> None:
        self._show_stt_status_dialog(
            f"음성 인식 실패: {message}",
            icon=QMessageBox.Icon.Warning,
        )
        self._schedule_stt_status_close(2500)
        if self._stt_context == "selection":
            self._update_selection_voice_button(False)
        self._show_mic_info("음성 인식 실패")
        self._stt_status_hide_timer.start(2500)

    def _start_mic_feedback(self) -> None:
        # 화면 하단 상태 표시 라벨을 즉시 노출해 사용자가 진행 상황을 인지하게 한다.
        self._stt_status_hide_timer.stop()
        self._stt_feedback_timer.stop()
        self._show_mic_info("음성 인식 중...")

    def _on_stt_feedback_tick(self) -> None:
        # 1초마다 메시지를 갱신해 장시간 대기 시에도 피드백을 유지한다.
        self._stt_feedback_timer.stop()
        self._show_mic_info("음성 인식 중...")

    def _show_mic_info(self, text: str) -> None:
        if self.mic_info_label is None:
            return
        self.mic_info_label.setText(text)
        self.mic_info_label.setVisible(True)

    def _hide_mic_info(self) -> None:
        if self.mic_info_label is None:
            return
        self.mic_info_label.setText("")
        self.mic_info_label.setVisible(False)

    def request_total_products(self) -> None:
        """전체 상품 목록을 조회해 초기 화면에 표시한다."""
        if self.service_client is None:
            QMessageBox.warning(
                self, "상품 로드 실패", "상품 서비스를 사용할 수 없습니다."
            )
            self.set_products([])
            self.refresh_product_grid()
            return
        user_id = self._ensure_user_identity()
        if not user_id:
            QMessageBox.warning(
                self,
                "상품 로드 실패",
                "사용자 정보를 확인할 수 없어 상품을 불러올 수 없습니다.",
            )
            self.set_products([])
            self.refresh_product_grid()
            return
        try:
            response = self.service_client.fetch_total_products(user_id)
        except MainServiceClientError as exc:
            QMessageBox.warning(
                self, "상품 로드 실패", f"전체 상품을 불러오지 못했습니다.\n{exc}"
            )
            self.set_products([])
            self.refresh_product_grid()
            return
        if not response:
            QMessageBox.warning(
                self, "상품 로드 실패", "서버에서 전체 상품 응답을 받지 못했습니다."
            )
            self.set_products([])
            self.refresh_product_grid()
            return
        if not response.get("result"):
            message = response.get("message") or "전체 상품을 가져오지 못했습니다."
            QMessageBox.warning(self, "상품 로드 실패", message)
            self.set_products([])
            self.refresh_product_grid()
            return
        data = response.get("data") or {}
        entries = data.get("products") or []
        products = self._convert_total_products(entries)
        if not products:
            QMessageBox.information(self, "상품 로드 안내", "표시할 상품이 없습니다.")
            self.empty_products_message = "표시할 상품이 없습니다."
            self.set_products([])
            self.refresh_product_grid()
            return
        self.empty_products_message = self.default_empty_products_message
        self.set_products(products)
        self.refresh_product_grid()

    def request_product_search(self, query: str) -> None:
        """사용자 입력과 필터 조건으로 상품 검색을 수행한다."""
        # 서비스 클라이언트가 없다면 네트워크 요청 자체가 불가능하므로 즉시 반환한다.
        if self.service_client is None:
            return
        user_id = self._ensure_user_identity()
        if not user_id:
            QMessageBox.warning(self, "검색 실패", "사용자 정보를 확인할 수 없습니다.")
            return
        # 사용자 정보를 기반으로 필터를 구성하지 않으면 개인화된 검색 조건이 적용되지 않는다.
        allergy_filter, vegan_flag = self._build_search_filter()
        # 검색 입력 위젯 참조를 보관하지 않으면 이후에 상태를 복원할 수 없다.
        search_widget = self.search_input
        # 검색 버튼 상태를 추적하지 않으면 비활성화 후 복구할 수 없다.
        search_button = getattr(self, "search_button", None)
        # 위젯 존재 여부를 확인하지 않으면 None에 접근하면서 예외가 발생한다.
        if search_widget is not None:
            # 요청 동안 입력을 막지 않으면 사용자가 연타하여 중복 요청을 발생시킬 수 있다.
            search_widget.setEnabled(False)
        if search_button is not None:
            # 응답을 기다리는 동안 버튼을 비활성화하지 않으면 반복 클릭으로 중복 요청이 발생한다.
            search_button.setEnabled(False)
        # 네트워크 예외를 처리하지 않으면 오류가 발생할 때 애플리케이션이 그대로 종료된다.
        response = None
        try:
            self._show_stt_status_dialog(
                f"인식된 문장: {query}\n     요청한 음성으로 검색 중...",
                icon=QMessageBox.Icon.Information,
            )
            # 명세에 맞춘 검색 요청을 호출하지 않으면 서버로부터 상품 목록을 받을 수 없다.
            response = self.service_client.search_products(
                user_id=user_id,
                query=query,
                allergy_filter=allergy_filter,
                is_vegan=vegan_flag,
            )
        except MainServiceClientError as exc:
            self._close_stt_status_dialog()
            # 오류 알림을 하지 않으면 사용자가 검색 실패 원인을 알 수 없다.
            QMessageBox.warning(
                self, "검색 실패", f"상품 검색 중 오류가 발생했습니다.\n{exc}"
            )
            # 실패 시 목록을 초기화하지 않으면 사용자가 최신 상태를 확인하기 어렵다.
            self.empty_products_message = "상품을 불러오지 못했습니다."
            self.set_products([])
            # 상품 목록을 다시 그리지 않으면 기존 화면이 갱신되지 않는다.
            self.refresh_product_grid()
            return
        finally:
            # 요청이 끝난 뒤 입력을 다시 활성화하지 않으면 사용자가 이후 검색을 할 수 없다.
            if search_widget is not None:
                search_widget.setEnabled(True)
            if search_button is not None:
                # 버튼을 다시 활성화하지 않으면 사용자가 추가 검색을 수행할 수 없다.
                search_button.setEnabled(True)
        self._close_stt_status_dialog()
        # 응답이 비어 있으면 이후 처리에서 KeyError가 발생할 수 있으므로 여기서 중단한다.
        if not response:
            QMessageBox.warning(self, "검색 실패", "서버에서 응답을 받지 못했습니다.")
            self.empty_products_message = "상품을 불러오지 못했습니다."
            self.set_products([])
            self.refresh_product_grid()
            return
        # result 플래그를 확인하지 않으면 서버가 실패를 알린 경우에도 잘못된 데이터를 사용할 수 있다.
        if not response.get("result"):
            # 서버가 전달한 메시지를 표시하지 않으면 사용자가 실패 이유를 확인할 수 없다.
            message = response.get("message") or "상품을 불러오지 못했습니다."
            QMessageBox.warning(self, "검색 실패", message)
            self.empty_products_message = "상품을 불러오지 못했습니다."
            self.set_products([])
            self.refresh_product_grid()
            return
        # 데이터 섹션을 추출하지 않으면 실제 상품 목록에 접근할 수 없다.
        data = response.get("data") or {}
        # 상품 배열이 비어 있을 수 있으므로 안전하게 기본값을 사용한다.
        product_entries = data.get("products") or []
        # 응답을 도메인 객체로 변환하지 않으면 UI 카드가 필요한 속성을 읽어올 수 없다.
        products = self._convert_search_results(product_entries)
        # 검색 결과가 비어 있으면 사용자에게 안내하고 그리드를 비워야 혼란이 없다.
        if not products:
            self.empty_products_message = "조건에 맞는 상품이 없습니다."
            self.set_products([])
            self.refresh_product_grid()
            return
        # 변환된 상품을 상태에 반영하지 않으면 UI가 최신 정보를 표시하지 못한다.
        self.set_products(products)
        # 상품 목록을 다시 렌더링하지 않으면 화면에 여전히 이전 검색 결과가 남아 있다.
        self.refresh_product_grid()

    def _build_search_filter(self) -> tuple[dict[str, bool], bool | None]:
        """사용자 환경설정으로 검색 필터를 구성한다."""
        # 사용자 정보가 비어 있으면 알레르기 필터를 구성할 수 없으므로 빈 딕셔너리를 준비한다.
        raw_allergy = (
            self.user_info.get("allergy_info")
            if isinstance(self.user_info, dict)
            else {}
        )
        # 필터 값을 누적할 새로운 딕셔너리를 만들지 않으면 원본 데이터를 직접 수정하게 된다.
        normalized_allergy: dict[str, bool] = {}
        # 딕셔너리가 아닐 경우 순회가 불가능하므로 타입을 확인한다.
        if isinstance(raw_allergy, dict):
            # 각 항목을 순회하지 않으면 개별 알레르기 정보가 필터에 포함되지 않는다.
            for key, value in raw_allergy.items():
                # 문자열 키로 변환하지 않으면 Qt JSON 직렬화 시 타입 불일치가 생길 수 있다.
                normalized_key = str(key)
                # 값이 불리언이 아니면 명세와 어긋나므로 bool()로 강제한다.
                normalized_allergy[normalized_key] = bool(value)
        # 사용자 정보에 비건 여부가 없다면 None을 반환해 서버 기본값을 사용하도록 한다.
        vegan_value = (
            self.user_info.get("is_vegan") if isinstance(self.user_info, dict) else None
        )
        # 비건 값이 None이면 두 번째 항목으로 None을 넘겨 서버가 기본 동작을 따르도록 한다.
        if vegan_value is None:
            return normalized_allergy, None
        # bool()로 강제하지 않으면 0과 1 같은 값이 그대로 전달되어 혼란을 줄 수 있다.
        return normalized_allergy, bool(vegan_value)

    def _resolve_product_image(self, product_id: int, name: str) -> Path:
        """상품 이름이나 ID로 이미지 파일 경로를 결정한다."""
        # 이름 기반 매핑이 있으면 우선적으로 사용한다. 상품 ID가 재사용되더라도 이름은 정확하다는 가정이다.
        normalized_name = (name or "").strip()
        if normalized_name:
            name_path = self.PRODUCT_IMAGE_BY_NAME.get(normalized_name)
            if name_path and name_path.exists():
                return name_path
        # 이름으로 찾지 못했다면 상품 ID 기반 매핑을 시도한다.
        mapped_path = self.PRODUCT_IMAGE_BY_ID.get(product_id)
        if mapped_path and mapped_path.exists():
            return mapped_path
        # 매핑 결과가 없거나 파일이 없으면 기본 이미지를 사용한다.
        return ProductCard.FALLBACK_IMAGE

    def _parse_allergy_info(
        self,
        entry: dict[str, object],
        *,
        allergy_info_id: int,
    ) -> AllergyInfoData | None:
        """상품 항목에 포함된 알레르기 정보를 안전하게 추출한다."""
        # 알러지 정보가 딕셔너리 형태가 아니라면 안전하게 None으로 처리한다.
        raw_info = entry.get("allergy_info")
        if not isinstance(raw_info, dict):
            return None

        # bool()로 강제하지 않으면 0, 1과 같은 값이 그대로 남는다.
        def _flag(name: str) -> bool:
            return bool(raw_info.get(name))

        return AllergyInfoData(
            allergy_info_id=allergy_info_id,
            nuts=_flag("nuts"),
            milk=_flag("milk"),
            seafood=_flag("seafood"),
            soy=_flag("soy"),
            peach=_flag("peach"),
            gluten=_flag("gluten"),
            eggs=_flag("eggs"),
        )

    def _convert_search_results(
        self, entries: list[dict[str, object]]
    ) -> list[ProductData]:
        """검색 결과 응답을 UI 표현에 맞는 데이터 모델로 변환한다."""
        # 결과를 누적할 리스트가 없으면 변환된 상품을 반환할 수 없다.
        products: list[ProductData] = []
        # 이미지 경로 계산은 전용 헬퍼로 위임해 중복을 줄인다.

        # 안전한 정수 변환 함수를 정의하지 않으면 잘못된 값이 들어왔을 때 예외로 루프가 중단된다.
        def to_int(value: object, default: int = 0) -> int:
            # 변환을 시도하지 않으면 문자열이나 None 타입이 그대로 남아 계산에서 오류가 난다.
            try:
                # 변환된 정수를 즉시 반환하지 않으면 호출부가 값을 사용할 수 없다.
                return int(value)
            except (TypeError, ValueError):
                # 변환 실패 시 기본값을 돌려주지 않으면 호출부에서 추가적인 방어 코드를 반복해야 한다.
                return default

        # 각각의 상품을 순회하지 않으면 리스트 전체를 변환할 수 없다.
        for entry in entries:
            # 항목이 딕셔너리가 아니면 필요한 키를 읽을 수 없어 건너뛴다.
            if not isinstance(entry, dict):
                continue
            # 상품 ID를 추출하지 않으면 장바구니 등 다른 기능에서 식별할 수 없다.
            product_id_value = entry.get("product_id")
            try:
                # ID를 정수로 만들지 않으면 데이터 클래스 생성 시 타입 오류가 난다.
                product_id = int(product_id_value)
            except (TypeError, ValueError):
                # 변환 실패 시 해당 항목을 건너뛰지 않으면 이후 로직이 예외로 중단된다.
                continue
            # 이름이 비어 있으면 카드에 빈 문자열이 표시되어 사용자에게 혼란을 준다.
            name = str(entry.get("name") or "")
            if not name:
                # 대체 이름을 제공하지 않으면 화면에서 해당 상품을 구분할 수 없다.
                name = f"상품 {product_id}"
            # 카테고리를 기본값으로 설정하지 않으면 None이 그대로 노출된다.
            category = str(entry.get("category") or "기타")
            # 가격을 정수로 변환하지 않으면 금액 표시에 문제가 생긴다.
            price = to_int(entry.get("price"), 0)
            # 할인율 없이는 할인 가격 계산이 불가능하므로 기본값 0을 사용한다.
            discount_rate = to_int(entry.get("discount_rate"), 0)
            # 알레르기 ID가 없으면 0으로 처리해 참조 오류를 막는다.
            allergy_info_id = to_int(entry.get("allergy_info_id"), 0)
            # 비건 여부를 bool로 강제하지 않으면 문자열 'false'가 그대로 표시될 수 있다.
            is_vegan_friendly = bool(entry.get("is_vegan_friendly"))
            # 섹션 ID가 없으면 0으로 지정해 UI에서 숫자 표시를 유지한다.
            section_id = to_int(entry.get("section_id"), 0)
            # 창고 ID가 없을 수 있으므로 기본값 0으로 설정한다.
            warehouse_id = to_int(entry.get("warehouse_id"), 0)
            # 길이 정보가 없다면 0으로 두어 치수 계산에서 오류를 피한다.
            length = to_int(entry.get("length"), 0)
            # 너비 역시 제공되지 않으면 0으로 기본 처리한다.
            width = to_int(entry.get("width"), 0)
            # 높이를 0으로 돌려두지 않으면 None 타입 곱셈에서 오류가 난다.
            height = to_int(entry.get("height"), 0)
            # 무게 정보가 누락되면 0으로 처리해 계산 시 에러를 방지한다.
            weight = to_int(entry.get("weight"), 0)
            # 깨지기 쉬운지 여부가 None이면 False로 간주하지 않으면 조건문에서 문제가 발생한다.
            fragile = bool(entry.get("fragile"))
            try:
                # 변환된 값을 데이터 클래스로 포장하지 않으면 UI 위젯이 활용할 수 없다.
                product = ProductData(
                    product_id=product_id,
                    name=name,
                    category=category,
                    price=price,
                    discount_rate=discount_rate,
                    allergy_info_id=allergy_info_id,
                    is_vegan_friendly=is_vegan_friendly,
                    section_id=section_id,
                    warehouse_id=warehouse_id,
                    length=length,
                    width=width,
                    height=height,
                    weight=weight,
                    fragile=fragile,
                    image_path=self._resolve_product_image(product_id, name),
                    allergy_info=self._parse_allergy_info(
                        entry,
                        allergy_info_id=allergy_info_id,
                    ),
                )
            except TypeError:
                # 필수 필드가 누락된 경우 해당 상품만 건너뛰어 전체 처리를 계속한다.
                continue
            # 누락 없이 생성된 상품만 리스트에 추가한다.
            products.append(product)
        # 변환된 전체 목록을 반환하지 않으면 호출자가 결과를 사용할 수 없다.
        return products

    def _convert_total_products(
        self, entries: list[dict[str, object]]
    ) -> list[ProductData]:
        """전체 상품 목록 응답을 ProductData 목록으로 변환한다."""
        # 전체 상품 응답이 비어 있으면 빈 리스트를 반환해야 이후 로직에서 목업 데이터를 사용할 수 있다.
        products: list[ProductData] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                product_id = int(entry.get("product_id"))
            except (TypeError, ValueError):
                continue
            name = str(entry.get("name") or f"상품 {product_id}")
            category = str(entry.get("category") or "기타")
            price = int(entry.get("price") or 0)
            discount_rate = int(entry.get("discount_rate") or 0)
            is_vegan = bool(entry.get("is_vegan_friendly"))
            # total_product 응답에는 allergy_info_id, section_id 등이 없으므로 기본값을 채워 넣는다.
            product = ProductData(
                product_id=product_id,
                name=name,
                category=category,
                price=price,
                discount_rate=discount_rate,
                allergy_info_id=0,
                is_vegan_friendly=is_vegan,
                section_id=int(entry.get("section_id") or 0),
                warehouse_id=int(entry.get("warehouse_id") or 0),
                length=int(entry.get("length") or 0),
                width=int(entry.get("width") or 0),
                height=int(entry.get("height") or 0),
                weight=int(entry.get("weight") or 0),
                fragile=bool(entry.get("fragile")),
                image_path=self._resolve_product_image(product_id, name),
                allergy_info=self._parse_allergy_info(
                    entry,
                    allergy_info_id=0,
                ),
            )
            products.append(product)
        return products

    def setup_cart_section(self):
        """장바구니 영역의 컨테이너와 토글 버튼을 구성한다."""
        self.cart_container = getattr(self.ui, "widget_3", None)
        self.cart_frame = getattr(self.ui, "cart_frame", None)
        self.cart_body = getattr(self.ui, "cart_body", None)
        self.cart_header = getattr(self.ui, "cart_header", None)
        self.cart_toggle_button = getattr(self.ui, "btn_chevron_up", None)
        self.product_scroll = getattr(self.ui, "scrollArea", None)

        if not self.cart_frame:
            return

        self.cart_container_layout = (
            self.cart_container.layout() if self.cart_container else None
        )
        if self.cart_container_layout is not None:
            margins = self.cart_container_layout.contentsMargins()
            self.cart_margin_left = margins.left()
            self.cart_margin_right = margins.right()
            self.cart_margin_bottom = margins.bottom()
            self.cart_margin_collapsed_top = margins.top()
            self.cart_margin_expanded_top = self.cart_margin_collapsed_top + max(
                8, self.cart_margin_expanded_top
            )

        self.cart_frame.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )

        if self.cart_body:
            self.cart_body.hide()

        if self.cart_toggle_button is not None:
            icon_path = (
                Path(__file__).resolve().parent / "icons" / "chevron-up-circle.svg"
            )
            if icon_path.exists():
                base_pixmap = QPixmap(str(icon_path))
                if not base_pixmap.isNull():
                    self.cart_icon_up = QIcon(base_pixmap)
                    rotated_pixmap = base_pixmap.transformed(QTransform().rotate(180))
                    self.cart_icon_down = QIcon(rotated_pixmap)
                    self.cart_toggle_button.setIcon(self.cart_icon_up)
            self.cart_toggle_button.clicked.connect(self.on_cart_toggle_clicked)
            self.cart_toggle_button.setFlat(True)
            self.cart_toggle_button.setIconSize(QtCore.QSize(28, 28))
            self.cart_toggle_button.setText("")
            self.cart_toggle_button.setToolTip("장바구니 펼치기")

        if self.product_scroll is not None:
            self.product_scroll.show()

        self.apply_cart_state()
        self.update_cart_empty_state()

    def setup_navigation(self):
        """상단 네비게이션과 사이드 스택 위젯을 초기화한다."""
        self.main_stack = getattr(self.ui, "stacked_content", None)
        self.side_stack = getattr(self.ui, "stack_side_bar", None)
        self.page_user = getattr(self.ui, "page_content_user", None)
        self.page_pick = getattr(self.ui, "page_content_pick", None)
        self.side_shop_page = getattr(self.ui, "side_pick_page", None)
        self.side_pick_filter_page = getattr(self.ui, "side_allergy_filter_page", None)
        self.shopping_button = getattr(self.ui, "btn_nav_shop", None)
        if self.shopping_button is None:
            self.shopping_button = getattr(self.ui, "toolButton_3", None)
        self.store_button = getattr(self.ui, "btn_nav_store", None)
        if self.store_button is None:
            self.store_button = getattr(self.ui, "toolButton_2", None)

        # QSS 스타일시트를 적용하여 세그먼트 버튼 디자인을 구현합니다.
        # 이 스타일은 'seg' 속성을 사용하여 각 버튼(왼쪽, 오른쪽)을 식별하고
        # :checked 상태에 따라 모양과 색상을 변경합니다.
        primary_color = COLORS["primary"]
        qss = f"""
        /* 공통 베이스: 회색 테두리, 글자 회색, 약간의 패딩 */
        QAbstractButton[seg="left"], QAbstractButton[seg="right"] {{
            border: 1px solid #D3D3D3;         /* 회색 보더 */
            background: #F2F2F2;               /* 비활성 회색 */
            color: #9AA0A6;                    /* 비활성 글자색 */
            padding: 2px 14px;
            font-weight: 600;
        }}

        /* 왼쪽 캡슐 */
        QAbstractButton[seg="left"] {{
            border-top-left-radius: 3px;
            border-bottom-left-radius: 3px;
            border-right: 0;                   /* 가운데 라인 제거 */
        }}

        /* 오른쪽 캡슐 */
        QAbstractButton[seg="right"] {{
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
        }}

        /* 체크(선택) 상태: 흰 배경 + 빨강 글자, 빨강 테두리 */
        QAbstractButton[seg="left"]:checked,
        QAbstractButton[seg="right"]:checked {{
            background: #FFFFFF;
            color: {primary_color};
            border-color: {primary_color}; /* 클릭된 버튼 테두리 빨강으로 변경 */
        }}

        /* hover 시 살짝 밝게 */
        QAbstractButton[seg="left"]:hover,
        QAbstractButton[seg="right"]:hover {{
            background: #FAFAFA;
        }}

        /* 포커스 윤곽선 없애기(원하면) */
        QAbstractButton[seg="left"]:focus,
        QAbstractButton[seg="right"]:focus {{
            outline: none;
        }}
        """
        self._segment_button_qss = qss
        self.setStyleSheet(qss)

        if self.shopping_button:
            self.shopping_button.setProperty("seg", "left")
            self.shopping_button.setText("쇼핑")
            self.shopping_button.setCheckable(True)
            self.shopping_button.setAutoRaise(False)
            self.shopping_button.setMinimumHeight(24)

        if self.store_button:
            self.store_button.setProperty("seg", "right")
            self.store_button.setText("매장")
            self.store_button.setCheckable(True)
            self.store_button.setAutoRaise(False)
            self.store_button.setMinimumHeight(24)

        if self.shopping_button or self.store_button:
            self.nav_group = QButtonGroup(self)
            self.nav_group.setExclusive(True)
            if self.shopping_button:
                self.nav_group.addButton(self.shopping_button)
            if self.store_button:
                self.nav_group.addButton(self.store_button)

        if self.shopping_button:
            self.shopping_button.clicked.connect(self.on_shopping_button_clicked)
        if self.store_button:
            self.store_button.clicked.connect(self.on_store_button_clicked)

        self.set_mode("shopping")
        self._apply_camera_toggle_styles()

    def _apply_camera_toggle_styles(self) -> None:
        """전면/로봇팔 전환 버튼에 세그먼트 스타일을 적용한다."""
        if self.front_view_button is None or self.arm_view_button is None:
            return
        self.front_view_button.setProperty("seg", "left")
        self.arm_view_button.setProperty("seg", "right")
        self.front_view_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.arm_view_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.front_view_button.setMinimumHeight(32)
        self.arm_view_button.setMinimumHeight(32)
        stylesheet = getattr(self, "_segment_button_qss", "")
        if stylesheet:
            self.setStyleSheet(stylesheet)

    def _setup_allergy_toggle(self) -> None:
        button = getattr(self.ui, "btn_allergy", None)
        self.allergy_sub_widget = getattr(self.ui, "widget_allergy_sub", None)
        if button is None or self.allergy_sub_widget is None:
            return
        self.allergy_toggle_button = button
        try:
            button.clicked.disconnect()
        except TypeError:
            pass
        button.setCheckable(True)
        button.setChecked(True)
        icon_path = Path(__file__).resolve().parent / "icons" / "fold.svg"
        if icon_path.exists():
            base_pixmap = QPixmap(str(icon_path))
            if not base_pixmap.isNull():
                self.allergy_icon_fold = QIcon(base_pixmap)
                rotated_pixmap = base_pixmap.transformed(QTransform().rotate(180))
                self.allergy_icon_fold_rotated = QIcon(rotated_pixmap)
                button.setIcon(self.allergy_icon_fold_rotated)
                button.setIconSize(QtCore.QSize(20, 20))
        button.setFlat(True)
        button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        button.setStyleSheet(
            "QPushButton {background-color: transparent; border: none;}"
            "QPushButton:checked {background-color: transparent; border: none;}"
            "QPushButton:hover {background-color: transparent; border: none;}"
            "QPushButton:pressed {background-color: transparent; border: none;}"
            "QPushButton:focus {background-color: transparent; border: none; outline: 0;}"
        )
        self._allergy_max_height = self.allergy_sub_widget.maximumHeight()
        if self._allergy_max_height <= 0 or self._allergy_max_height >= 16777215:
            size_hint = self.allergy_sub_widget.sizeHint()
            self._allergy_max_height = size_hint.height()
        if self._allergy_max_height <= 0:
            self._allergy_max_height = 16777215
        self.allergy_filters_expanded = True
        self._apply_allergy_toggle_state(expanded=self.allergy_filters_expanded)
        button.toggled.connect(self._on_allergy_toggle_clicked)

    def _setup_allergy_checkboxes(self) -> None:
        """알레르기 체크박스를 수집해 상태 동기화를 준비한다."""
        # 알러지 필터 체크박스를 동기화하지 않으면 상위와 하위 항목이 따로 움직인다.
        self.allergy_total_checkbox = getattr(self.ui, "cb_allergy_total", None)
        checkbox_names = {
            # 견과류 알러지 여부를 제어하는 체크박스 이름과 키 쌍
            "cb_nuts": "nuts",
            # 유제품 알러지 여부를 제어하는 체크박스 이름과 키 쌍
            "cb_milk": "milk",
            # 어폐류 알러지 여부를 제어하는 체크박스 이름과 키 쌍
            "cb_seafood": "seafood",
            # 대두/콩 알러지 여부를 제어하는 체크박스 이름과 키 쌍
            "cb_bean": "soy",
            # 복숭아 알러지 여부를 제어하는 체크박스 이름과 키 쌍
            "cb_peach": "peach",
            # 글루텐 알러지 여부를 제어하는 체크박스 이름과 키 쌍
            "cb_gluten": "gluten",
            # 계란 알러지 여부를 제어하는 체크박스 이름과 키 쌍
            "cb_egg": "eggs",
        }
        # 키별로 연결된 체크박스를 저장하는 매핑을 비워둔다.
        self.allergy_checkbox_map = {}
        if self.allergy_total_checkbox is not None:
            try:
                # 기존에 연결된 신호가 있다면 제거해 중복 연결을 방지한다.
                self.allergy_total_checkbox.stateChanged.disconnect(
                    self._on_allergy_total_state_changed
                )
            except TypeError:
                # 연결이 없으면 예외가 발생하므로 무시한다.
                pass
            # 상위 체크박스를 이진 상태로만 사용하도록 설정한다.
            self.allergy_total_checkbox.setTristate(False)
            # 상위 체크박스 상태가 변할 때 하위 항목을 동기화하기 위해 신호를 연결한다.
            self.allergy_total_checkbox.stateChanged.connect(
                self._on_allergy_total_state_changed
            )
        for name, key in checkbox_names.items():
            # 각 이름에 해당하는 체크박스를 UI에서 가져온다.
            checkbox = getattr(self.ui, name, None)
            if not isinstance(checkbox, QCheckBox):
                # 체크박스가 아니면 목록에 포함시키지 않는다.
                continue
            try:
                # 기존 신호 연결이 있다면 제거한다.
                checkbox.stateChanged.disconnect(self._on_allergy_child_state_changed)
            except TypeError:
                # 연결이 없으면 예외가 발생하므로 무시한다.
                pass
            # 하위 체크박스 상태 변화 시 상위 체크박스를 업데이트하도록 신호를 연결한다.
            checkbox.stateChanged.connect(self._on_allergy_child_state_changed)
            # 동기화 대상 목록에 체크박스를 추가한다.
            self.allergy_checkbox_map[key] = checkbox
        vegan_checkbox = getattr(self.ui, "cb_vegan", None)
        if isinstance(vegan_checkbox, QCheckBox):
            self.vegan_checkbox = vegan_checkbox
            try:
                vegan_checkbox.stateChanged.disconnect(self._on_vegan_state_changed)
            except TypeError:
                pass
            vegan_checkbox.stateChanged.connect(self._on_vegan_state_changed)
        if self.allergy_total_checkbox is not None and self.allergy_checkbox_map:
            # 하위 체크박스 상태를 기준으로 상위 상태를 재조정한다.
            self._sync_allergy_total_from_children()
        else:
            # 상위 또는 하위 체크박스가 없으면 기본 상태를 유지한다.
            self._sync_allergy_total_from_children()
        # 현재 체크 상태에 맞춰 초기 필터를 적용하지 않으면 UI와 목록이 불일치한다.
        self._apply_product_filters(refresh=False)

    def _apply_allergy_toggle_state(self, *, expanded: bool) -> None:
        """토글 상태에 맞춰 서브 위젯 표시 여부와 아이콘을 갱신한다."""
        if self.allergy_toggle_button is None or self.allergy_sub_widget is None:
            return
        self.allergy_filters_expanded = expanded
        if expanded:
            self.allergy_toggle_button.setToolTip("알러지 필터 접기")
            if self.allergy_icon_fold_rotated is not None:
                self.allergy_toggle_button.setIcon(self.allergy_icon_fold_rotated)
            self.allergy_sub_widget.setMaximumHeight(self._allergy_max_height)
            self.allergy_sub_widget.show()
            return
        self.allergy_toggle_button.setToolTip("알러지 필터 펼치기")
        if self.allergy_icon_fold is not None:
            self.allergy_toggle_button.setIcon(self.allergy_icon_fold)
        self.allergy_sub_widget.setMaximumHeight(0)
        self.allergy_sub_widget.hide()

    def _on_allergy_toggle_clicked(self, checked: bool) -> None:
        """토글 버튼 클릭에 대해 확장 상태를 반영한다."""
        if self.allergy_toggle_button is None or self.allergy_sub_widget is None:
            return
        self._apply_allergy_toggle_state(expanded=checked)

    def _on_allergy_total_state_changed(
        self, state: int | QtCore.Qt.CheckState
    ) -> None:
        """상위 체크박스 상태에 따라 하위 항목을 일괄 갱신한다."""
        # 하위 체크박스가 없으면 동기화를 진행할 수 없다.
        if not self.allergy_checkbox_map:
            return
        try:
            # PyQt 신호는 정수 또는 CheckState 열거형을 넘길 수 있으므로 공통 enum으로 변환한다.
            state_enum = QtCore.Qt.CheckState(state)
        except (ValueError, TypeError):
            # 변환에 실패하면 해제 상태로 취급해 불필요한 예외를 막는다.
            state_enum = QtCore.Qt.CheckState.Unchecked
        # 상위 체크박스가 체크 상태인지 확인한다.
        checked = state_enum == QtCore.Qt.CheckState.Checked
        # 상위 상태에 맞춰 모든 하위 체크박스를 일괄 설정한다.
        self._set_allergy_children_checked(checked=checked)
        # 하위 상태를 다시 읽어 상위 체크박스를 일관성 있게 유지한다.
        self._sync_allergy_total_from_children()
        # 상위 필터가 바뀌면 상품 목록도 즉시 갱신해야 한다.
        self._apply_product_filters()

    def _on_allergy_child_state_changed(self, _state: int) -> None:
        """하위 체크박스 변경 시 상위 상태와 필터를 다시 계산한다."""
        # 상위 체크박스가 없거나 하위 목록이 비어 있으면 처리하지 않는다.
        if self.allergy_total_checkbox is None or not self.allergy_checkbox_map:
            return
        # 어느 하위 항목이든 변경되면 상위 체크박스 상태를 다시 계산한다.
        self._sync_allergy_total_from_children()
        # 하위 필터가 바뀌었으므로 상품 목록을 재평가한다.
        self._apply_product_filters()

    def _on_vegan_state_changed(self, _state: int) -> None:
        """비건 필터 변경에 맞춰 검색 조건을 다시 적용한다."""
        # 비건 필터만 변경되어도 상품 목록을 다시 계산해야 한다.
        self._apply_product_filters()

    def _set_allergy_children_checked(self, *, checked: bool) -> None:
        """하위 알레르기 체크박스를 일괄적으로 설정한다."""
        # 신호 루프를 막기 위해 블록하면서 모든 하위 체크박스를 설정한다.
        for checkbox in self.allergy_checkbox_map.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def _sync_allergy_total_from_children(self) -> None:
        """하위 체크 상태를 검토해 상위 체크박스 표시를 동기화한다."""
        # 상위 체크박스나 하위 목록이 없으면 더 이상 동기화를 진행하지 않는다.
        if self.allergy_total_checkbox is None or not self.allergy_checkbox_map:
            return
        # 모든 하위 체크박스가 선택된 경우에만 상위 체크박스를 체크한다.
        all_checked = all(
            checkbox.isChecked() for checkbox in self.allergy_checkbox_map.values()
        )
        self.allergy_total_checkbox.blockSignals(True)
        self.allergy_total_checkbox.setChecked(all_checked)
        self.allergy_total_checkbox.blockSignals(False)

    def _apply_main_banner_image(self) -> None:
        # 상단 배너 이미지를 로드해 label_main_top에 표시한다.
        label = getattr(self.ui, "label_main_top", None)
        if not isinstance(label, QLabel):
            return
        banner_path = Path(__file__).resolve().parent / "image" / "main_top.png"
        if not banner_path.exists():
            return
        pixmap = QPixmap(str(banner_path))
        if pixmap.isNull():
            return
        label.setPixmap(pixmap)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def _style_main_banner_title(self) -> None:
        # 배너 제목 라벨을 굵은 빨간색 12pt로 통일한다.
        label = getattr(self.ui, "label_let_start", None)
        if not isinstance(label, QLabel):
            return
        font = label.font()
        font.setPointSize(16)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet("color: #ff0000;")

    def _style_do_create_label(self) -> None:
        # 주문담기 안내 라벨을 굵은 검정색 18pt로 설정한다.
        label = getattr(self.ui, "label_do_create", None)
        if not isinstance(label, QLabel):
            return
        font = label.font()
        font.setPointSize(16)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet("color: #000000;")

    def _apply_product_filters(self, *, refresh: bool = True) -> None:
        # 필터 적용 기준이 없으면 기존 상품 목록을 그대로 유지한다.
        base_products = list(self.all_products)
        vegan_checkbox = self.vegan_checkbox
        vegan_exclude = (
            isinstance(vegan_checkbox, QCheckBox) and not vegan_checkbox.isChecked()
        )
        disabled_allergies = {
            key
            for key, checkbox in self.allergy_checkbox_map.items()
            if isinstance(checkbox, QCheckBox) and not checkbox.isChecked()
        }
        filtered_products: list[ProductData] = []
        for product in base_products:
            if not self._product_matches_filters(
                product,
                disabled_allergies=disabled_allergies,
                vegan_exclude=vegan_exclude,
            ):
                continue
            filtered_products.append(product)
        self.products = filtered_products
        self.product_index = {
            product.product_id: product for product in filtered_products
        }
        if filtered_products:
            self.empty_products_message = self.default_empty_products_message
        elif base_products:
            self.empty_products_message = self.filtered_empty_message
        else:
            self.empty_products_message = self.default_empty_products_message
        self.current_columns = -1
        if refresh:
            self.refresh_product_grid()

    def _product_matches_filters(
        self,
        product: ProductData,
        *,
        disabled_allergies: set[str],
        vegan_exclude: bool,
    ) -> bool:
        # 비건 필터: 체크 해제되었을 때 비건 상품 제외
        if vegan_exclude and product.is_vegan_friendly:
            return False

        # 알러지 정보가 없는 상품은 기본적으로 허용
        allergy_info = product.allergy_info
        if allergy_info is None:
            return True

        # 해제된 알러지를 가진 상품은 제외
        for key in disabled_allergies:
            if getattr(allergy_info, key, False):
                return False

        return True

    def on_cart_toggle_clicked(self):
        if self.cart_toggle_button is None:
            return

        self.cart_expanded = not self.cart_expanded
        self.apply_cart_state()

    def on_shopping_button_clicked(self):
        self.set_mode("shopping")

    def on_store_button_clicked(self):
        self.set_mode("pick")

    def on_shop_end_clicked(self) -> None:
        self.on_logout_requested()

    def on_shop_continue_clicked(self) -> None:
        self.pick_flow_completed = False
        self.set_mode("shopping")
        if self.order_select_stack is not None and self.page_moving_view is not None:
            self.order_select_stack.setCurrentWidget(self.page_moving_view)
        if self.active_camera_type:
            self._update_video_buttons(self.active_camera_type)
        else:
            self._auto_start_front_camera_if_possible()

    def request_create_order(self) -> bool:
        if not getattr(self, "service_client", None):
            return False

        user_id = getattr(self, "current_user_id", "")
        if not user_id:
            QMessageBox.warning(self, "주문 생성 실패", "사용자 정보가 없습니다.")
            return False

        selected_items = [item for item in self.cart_items.values() if item.is_selected]
        if not selected_items:
            QMessageBox.warning(self, "주문 생성 실패", "선택된 상품이 없습니다.")
            return False

        selected_snapshot = [replace(item) for item in selected_items]
        total_amount = sum(item.total_price for item in selected_items)
        payment_method = "card"

        try:
            response = self.service_client.create_order(
                user_id=user_id,
                cart_items=selected_items,
                payment_method=payment_method,
                total_amount=total_amount,
            )
        except MainServiceClientError as exc:
            QMessageBox.warning(self, "주문 생성 실패", str(exc))
            return False

        if not response:
            QMessageBox.warning(self, "주문 생성 실패", "서버 응답이 없습니다.")
            return False

        if response.get("result"):
            order_data = response.get("data") or {}
            self.handle_order_created(selected_snapshot, order_data)
            self.clear_ordered_cart_items(selected_items)
            QMessageBox.information(
                self,
                "주문 생성 완료",
                response.get("message") or "주문이 생성되었습니다.",
            )
            return True

        QMessageBox.warning(
            self,
            "주문 생성 실패",
            response.get("message") or "주문 생성에 실패했습니다.",
        )
        return False

    def on_notification_received(self, payload: dict) -> None:
        """푸시 알림 수신 시 처리."""
        msg_type = payload.get("type")
        if msg_type == "robot_moving_notification":
            self.handle_robot_moving_notification(payload)
        elif msg_type == "robot_arrived_notification":
            self.handle_robot_arrived_notification(payload)
        elif msg_type == "picking_complete_notification":
            self.handle_picking_complete_notification(payload)
        elif msg_type == "product_selection_start":
            self.handle_product_selection_start(payload)
        elif msg_type == "cart_update_notification":
            self.handle_cart_update_notification(payload)

    def on_notification_error(self, message: str) -> None:
        """알림 수신 중 오류 발생 시 사용자에게 전달."""
        # TODO: GUI에 상태 표시 위젯을 추가해 오류 메시지를 노출한다.
        print(f"알림 수신 오류: {message}")

    def handle_order_created(
        self,
        ordered_items: list[CartItemData],
        order_data: dict,
    ) -> None:
        self.pick_flow_completed = False
        self._footer_locked = False
        self._ensure_notification_listener()
        auto_select_map = self._build_order_auto_select_map(order_data)
        remote_items, auto_items = self.categorize_cart_items(
            ordered_items,
            auto_select_map,
        )
        self.remote_selection_items = remote_items
        self.auto_selection_items = auto_items
        self.selection_item_states.clear()

        remote_widget = getattr(self.ui, "widget_remote_select_list", None)
        auto_widget = getattr(self.ui, "widget_auto_select_list", None)
        self.populate_selection_list(remote_widget, remote_items, "원격 선택 대기")
        self.populate_selection_list(auto_widget, auto_items, "자동 선택 대기")
        self._update_selection_state_label()

        self.current_order_id = order_data.get("order_id")
        self.current_robot_id = order_data.get("robot_id")
        self._auto_start_front_camera_if_possible()

        status_label = getattr(self.ui, "label_12", None)
        if status_label is not None:
            if self.current_robot_id is not None:
                status_label.setText(f"로봇 {self.current_robot_id} 이동중")
            else:
                status_label.setText("로봇 id ")

    def clear_ordered_cart_items(self, items: list[CartItemData]) -> None:
        for item in items:
            if item.product_id in self.cart_items:
                del self.cart_items[item.product_id]
            self.remove_cart_widget(item.product_id)
        self.update_cart_summary()
        self.sync_select_all_state()

    def handle_robot_moving_notification(self, payload: dict) -> None:
        """로봇 이동 알림에 따라 상태 텍스트를 갱신한다."""
        if not payload.get("result"):
            return
        if self.current_order_id is None:
            return
        data = payload.get("data") or {}
        order_id = data.get("order_id")
        if self.current_order_id >= 0 and order_id not in (None, self.current_order_id):
            return
        robot_id = data.get("robot_id")
        if robot_id is not None:
            self.current_robot_id = robot_id
            self._auto_start_front_camera_if_possible()
        destination = data.get("destination")
        message_text = payload.get("message") or "로봇 이동중"
        friendly_destination = self._get_friendly_section_name(destination)
        if friendly_destination:
            formatted = f"현재 <b>{friendly_destination}</b> 매대로 이동 중입니다."
        elif destination:
            formatted = f"{message_text} : {destination}"
        else:
            formatted = message_text
        status_label = getattr(self.ui, "label_12", None)
        if not self.pick_flow_completed and status_label is not None:
            status_label.setText(formatted)
        self._update_footer_label(formatted, context="robot_moving_notification")

        print(f"[알림] {formatted}")
        self._show_moving_page()

    def handle_robot_arrived_notification(self, payload: dict) -> None:
        """로봇 도착 알림에 따라 상태 텍스트를 갱신한다."""
        if not payload.get("result"):
            return
        if self.current_order_id is None:
            return
        data = payload.get("data") or {}
        order_id = data.get("order_id")
        if self.current_order_id >= 0 and order_id not in (None, self.current_order_id):
            return
        robot_id = data.get("robot_id")
        if robot_id is not None:
            self.current_robot_id = robot_id
            self._auto_start_front_camera_if_possible()
            # 도착 시점에는 즉시 팔 카메라로 전환해 상품 선택 화면을 구성한다.
            self._ensure_arm_camera_view()

        section_id = data.get("section_id")
        location_id = data.get("location_id")
        if isinstance(section_id, int) and section_id >= 0:
            location_text = f"SECTION_{section_id}"
        elif location_id is not None:
            location_text = f"LOCATION_{location_id}"
        else:
            location_text = None

        message_text = payload.get("message") or "로봇 도착"
        friendly_destination = self._get_friendly_section_name(location_text)
        if friendly_destination:
            formatted = f"<b>{friendly_destination}</b> 매대에 도착했습니다."
        elif location_text:
            formatted = f"{message_text} : {location_text}"
        else:
            formatted = message_text

        status_label = getattr(self.ui, "label_12", None)
        if not self.pick_flow_completed and status_label is not None:
            status_label.setText(formatted)
        self._update_footer_label(formatted, context="robot_arrived_notification")
        print(f"[알림] {formatted}")
        self._show_moving_page()

    def handle_picking_complete_notification(self, payload: dict) -> None:
        """모든 상품 담기 완료 알림을 처리한다."""
        if not payload.get("result"):
            return
        if self.current_order_id is None:
            return
        data = payload.get("data") or {}
        order_id = data.get("order_id")
        if self.current_order_id >= 0 and order_id not in (None, self.current_order_id):
            return
        robot_id = data.get("robot_id")
        if robot_id is not None:
            self.current_robot_id = robot_id

        message_text = payload.get("message") or "모든 상품 담기가 완료되었습니다"
        status_label = getattr(self.ui, "label_12", None)
        if status_label is not None:
            status_label.setText(message_text)
        self._update_footer_label(
            message_text,
            context="picking_complete_notification",
            force=True,
        )
        self._footer_locked = True
        print(f"[알림] {message_text}")
        self.pick_flow_completed = True
        self._mark_all_selection_completed()
        self._show_pick_complete_page()

    def handle_product_selection_start(self, payload: dict) -> None:
        """상품 선택 시작 알림을 처리한다."""
        if not payload.get("result"):
            return
        data = payload.get("data") or {}
        order_id = data.get("order_id")
        order_id_value: int | None = None
        if order_id is not None:
            try:
                order_id_value = int(order_id)
            except (TypeError, ValueError):
                order_id_value = None
            if (
                order_id_value is not None
                and self.current_order_id is not None
                and self.current_order_id >= 0
                and self.current_order_id not in (order_id_value,)
            ):
                return
            if order_id_value is not None:
                self.current_order_id = order_id_value
        if self.current_order_id is None:
            return
        robot_id = data.get("robot_id")
        if robot_id is not None:
            self.current_robot_id = robot_id

        raw_products = data.get("products") or []
        self.selection_options = []
        self.selection_options_origin = "service"
        for product in raw_products:
            if not isinstance(product, dict):
                continue
            product_id = product.get("product_id")
            bbox_number = product.get("bbox_number")
            try:
                product_id_value = int(product_id)
                bbox_value = int(bbox_number)
            except (TypeError, ValueError):
                continue
            name = str(product.get("name") or f"상품 {product_id_value}")
            self.selection_options.append(
                {
                    "product_id": product_id_value,
                    "bbox_number": bbox_value,
                    "name": name,
                }
            )
        self.selection_selected_index = None
        if self.select_title_label is not None:
            self.select_title_label.setText("원하는 상품을 선택해주세요.")

        product_names: list[str] = []
        for product in self.selection_options:
            name = product.get("name")
            if name:
                product_names.append(str(name))
        if product_names:
            products_text = ", ".join(product_names[:3])
            if len(product_names) > 3:
                products_text += " 외"
        else:
            products_text = None

        message_text = payload.get("message") or "상품 선택을 시작합니다"
        if products_text:
            formatted = f"{message_text} ({products_text})"
        else:
            formatted = message_text

        status_label = getattr(self.ui, "label_12", None)
        if status_label is not None:
            status_label.setText(formatted)
        self._update_footer_label(formatted, context="product_selection_start")
        self.populate_selection_buttons(self.selection_options)
        if self.order_select_stack is not None:
            target_page = (
                self.page_select_product
                if self.selection_options and self.page_select_product is not None
                else self.page_moving_view
            )
            if target_page is not None:
                self.order_select_stack.setCurrentWidget(target_page)
        print(f"[알림] {formatted}")
        self._update_selection_state_label()

    def handle_cart_update_notification(self, payload: dict) -> None:
        """장바구니 담기 알림을 처리한다."""
        if not payload.get("result"):
            return
        if self.current_order_id is None:
            return
        data = payload.get("data") or {}
        order_id = data.get("order_id")
        if self.current_order_id >= 0 and order_id not in (None, self.current_order_id):
            return
        robot_id = data.get("robot_id")
        if robot_id is not None:
            self.current_robot_id = robot_id

        action = data.get("action")
        product = data.get("product") or {}
        product_name = product.get("name")
        quantity = product.get("quantity")
        product_id = product.get("product_id")
        try:
            quantity_value = int(quantity)
        except (TypeError, ValueError):
            quantity_value = 0
        try:
            product_id_value = int(product_id)
        except (TypeError, ValueError):
            product_id_value = None

        if action == "add":
            default_message = "상품이 장바구니에 담겼습니다"
            if product_id_value is not None and quantity_value:
                self._update_selection_progress(product_id_value, quantity_value)
        elif action == "remove":
            default_message = "상품이 장바구니에서 제거되었습니다"
            if product_id_value is not None and quantity_value:
                self._update_selection_progress(product_id_value, -quantity_value)
        else:
            default_message = "장바구니가 갱신되었습니다"
        message_text = payload.get("message") or default_message

        details: list[str] = []
        if product_name:
            details.append(str(product_name))
        if quantity is not None:
            details.append(f"x{quantity}")
        if details:
            formatted = f"{message_text} ({' '.join(details)})"
        else:
            formatted = message_text

        status_label = getattr(self.ui, "label_12", None)
        if not self.pick_flow_completed and status_label is not None:
            status_label.setText(formatted)
        self._update_footer_label(formatted, context="cart_update_notification")
        print(f"[알림] {formatted}")
        self._show_moving_page()

    def _ensure_notification_listener(self) -> None:
        """주문 생성 시 알림 리스너를 시작한다."""
        user_id = self._ensure_user_identity()
        # 로그인 시 기억해 둔 비밀번호를 꺼내 알림 채널 인증에 재사용한다.
        password = ""
        if isinstance(self.user_info, dict):
            raw_password = self.user_info.get("password")
            if raw_password:
                password = str(raw_password)
        if self.notification_client is not None:
            # 이미 생성된 수신 스레드에는 최신 자격 증명을 갱신해 준다.
            self.notification_client.update_credentials(user_id, password)
            if not self.notification_client.isRunning():
                self.notification_client.start()
            return
        self.notification_client = AppNotificationClient(
            config=self.service_client.config,
            user_id=user_id,
            password=password,
        )
        self.notification_client.notification_received.connect(
            self.on_notification_received
        )
        self.notification_client.connection_error.connect(self.on_notification_error)
        self.notification_client.start()

    def _setup_refresh_logo(self) -> None:
        # 로고를 클릭 가능하게 만들어 전체 상품 목록을 즉시 새로 고칠 수 있도록 구성한다.
        top_layout = getattr(self.ui, "horizontalLayout", None)
        original_label = getattr(self.ui, "label_main_logo", None)
        if top_layout is None or original_label is None:
            return
        parent_widget = original_label.parent()
        existing_index = top_layout.indexOf(original_label)
        if existing_index < 0:
            existing_index = 0
        if isinstance(original_label, ClickableLabel):
            logo_label = original_label
        else:
            top_layout.removeWidget(original_label)
            original_label.hide()
            original_label.deleteLater()
            logo_label = ClickableLabel(parent_widget)
            logo_label.setObjectName("label_main_logo")
            top_layout.insertWidget(
                existing_index,
                logo_label,
                0,
                QtCore.Qt.AlignmentFlag.AlignLeft,
            )
            self.ui.label_main_logo = logo_label
        logo_label.setToolTip("전체 상품 새로고침")
        pixmap = self._render_logo_pixmap(QtCore.QSize(100, 30))
        if pixmap is not None:
            logo_label.setPixmap(pixmap)
            logo_label.setMinimumSize(pixmap.size())
        else:
            logo_label.setText("Shopee")
            logo_label.setStyleSheet(
                "font-size: 20px; font-weight: 600; color: #ff4649;"
            )
        logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        logo_label.clicked.connect(self._on_logo_refresh_clicked)
        self.refresh_logo_label = logo_label

    def _render_logo_pixmap(self, target_size: QtCore.QSize) -> QPixmap | None:
        # SVG 로고를 원하는 크기로 렌더링해 픽스맵으로 변환한다.
        logo_path = Path(__file__).resolve().parent / "image" / "logo.svg"
        if not logo_path.exists():
            return None
        renderer = QSvgRenderer(str(logo_path))
        if not renderer.isValid():
            return None
        pixmap = QPixmap(target_size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return pixmap

    def _on_logo_refresh_clicked(self) -> None:
        # 로고 클릭 시 전체 상품 목록을 즉시 재조회한다.
        self.request_total_products()

    def _initialize_selection_grid(self) -> None:
        if self.selection_grid is None:
            return
        while self.selection_grid.count():
            item = self.selection_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for column in range(5):
            self.selection_grid.setColumnStretch(column, 1)

    def populate_selection_buttons(self, products: list[dict[str, Any]]) -> None:
        if self.selection_grid is None:
            return
        while self.selection_grid.count():
            item = self.selection_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if isinstance(widget, QPushButton):
                    self.selection_button_group.removeButton(widget)
                widget.deleteLater()
        self.selection_buttons.clear()

        parent = (
            self.selection_container if self.selection_container is not None else self
        )
        for index, product in enumerate(products):
            button = QPushButton(parent)
            button.setCheckable(True)
            button.setFixedSize(123, 36)
            bbox_value = product.get("bbox_number")
            raw_label = str(product.get("label") or "").strip()
            raw_name = product.get("name")
            if isinstance(raw_name, str):
                name_text = raw_name.strip()
            elif raw_name is not None:
                name_text = str(raw_name)
            else:
                name_text = ""
            if not name_text:
                product_id_value = product.get("product_id")
                if product_id_value is not None:
                    name_text = str(product_id_value)
            display_label = raw_label
            if not display_label:
                base_name = name_text or f"선택지{index + 1}"
                try:
                    bbox_number_text = (
                        f"{int(bbox_value)}. " if bbox_value is not None else ""
                    )
                except (TypeError, ValueError):
                    bbox_number_text = ""
                display_label = f"{bbox_number_text}{base_name}".strip()
                if not display_label:
                    display_label = f"선택지{index + 1}"
            tooltip_text = name_text or display_label
            button.setText(display_label)
            button.setToolTip(tooltip_text)
            button.setProperty("option_index", index)
            self._apply_selection_button_style(button)
            self.selection_button_group.addButton(button)
            row, column = divmod(index, 5)
            self.selection_grid.addWidget(button, row, column)
            self.selection_buttons.append(button)
        for column in range(5):
            self.selection_grid.setColumnStretch(column, 1)
        if self.select_done_button is not None:
            self.select_done_button.setEnabled(bool(products))
        if self.auto_pick_button is not None:
            self.auto_pick_button.setEnabled(bool(products))
        if products:
            self._ensure_arm_camera_view()
        if self.selection_buttons:
            self.selection_buttons[0].setChecked(True)
            self.selection_selected_index = 0
        else:
            self.selection_selected_index = None
        self._refresh_bbox_highlight()

    def _apply_selection_button_style(self, button: QPushButton) -> None:
        # 선택 토글 버튼은 상태별 색상을 명확히 구분한다.
        primary_color = COLORS["primary"]
        neutral_background = COLORS["gray_light"]
        button.setStyleSheet(
            f"""
                QPushButton {{
                    background-color: {neutral_background};
                    color: #000000;
                    border: 1px solid transparent;
                    border-radius: 3px;
                    padding: 6px 12px;
                }}
                QPushButton:checked {{
                    background-color: #ffffff;
                    color: {primary_color};
                    border: 1px solid {primary_color};
                }}
            """
        )

    def _apply_pick_bottom_sheet_frame_style(self, frame: QWidget) -> None:
        # 픽업 바텀시트 프레임의 상단 모서리를 둥글게 처리한다.
        object_name = frame.objectName() or "frame_pick_bottom_sheet"
        frame.setObjectName(object_name)
        frame.setStyleSheet(
            f"""
                QFrame#{object_name} {{
                    background-color: #ffffff;
                    border: 1px solid #000000;
                    border-top-left-radius: 15px;
                    border-top-right-radius: 15px;
                    border-bottom-left-radius: 0px;
                    border-bottom-right-radius: 0px;
                }}
            """
        )

    def _apply_shop_end_button_style(self, button: QPushButton) -> None:
        # 쇼핑 종료 버튼은 강조 색상 테두리와 텍스트 대비를 유지한다.
        primary_color = COLORS["primary"]
        button.setStyleSheet(
            f"""
                QPushButton {{
                    background-color: #ffffff;
                    color: {primary_color};
                    border: 1px solid {primary_color};
                    border-radius: 3px;
                    padding: 8px 20px;
                }}
                QPushButton:pressed {{
                    background-color: #ffffff;
                }}
            """
        )

    def _apply_shop_continue_button_style(self, button: QPushButton) -> None:
        # 쇼핑 계속 버튼은 기본 색상으로 강조한다.
        primary_color = COLORS["primary"]
        primary_dark = COLORS.get("primary_dark", primary_color)
        button.setStyleSheet(
            f"""
                QPushButton {{
                    background-color: {primary_color};
                    color: #ffffff;
                    border: none;
                    border-radius: 3px;
                    padding: 8px 20px;
                }}
                QPushButton:pressed {{
                    background-color: {primary_dark};
                }}
            """
        )

    def _init_video_views(self) -> None:
        if self.gv_front is not None:
            self.front_scene = QGraphicsScene(self.gv_front)
            self.gv_front.setScene(self.front_scene)
            self.gv_front.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            self.gv_front.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.gv_front.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.front_item = self.front_scene.addPixmap(QPixmap())
        if self.gv_arm is not None:
            self.arm_scene = QGraphicsScene(self.gv_arm)
            self.gv_arm.setScene(self.arm_scene)
            self.gv_arm.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            self.gv_arm.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.gv_arm.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.arm_item = self.arm_scene.addPixmap(QPixmap())
        if self.video_stack is not None and self.page_front is not None:
            self.video_stack.setCurrentWidget(self.page_front)
        self._update_video_buttons(None)

    def _init_map_view(self) -> None:
        if self.map_view is None:
            return
        self.map_scene = QGraphicsScene(self.map_view)
        self.map_view.setScene(self.map_scene)
        self.map_view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.map_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        pixmap, resolution, origin = self._load_map_pixmap()
        self.map_pixmap_item = self.map_scene.addPixmap(pixmap)
        self.map_pixmap_item.setZValue(0)
        scene_rect = self.map_pixmap_item.boundingRect()
        self.map_scene.setSceneRect(scene_rect)
        self._fit_map_to_view()
        self.map_resolution = resolution
        self.map_origin = (origin[0], origin[1])
        self.map_image_size = (pixmap.width(), pixmap.height())
        robot_item, heading_item = self._create_robot_graphics()
        robot_item.setZValue(10)
        robot_item.setVisible(False)
        self.map_scene.addItem(robot_item)
        if heading_item is not None:
            heading_item.setZValue(11)
            heading_item.setVisible(False)
        self.map_robot_item = robot_item
        self.map_heading_item = heading_item
        if self.SHOW_ROBOT_LABEL:
            label_item = QGraphicsSimpleTextItem("")
            label_font = label_item.font()
            label_font.setPointSize(10)
            label_item.setFont(label_font)
            label_item.setBrush(QBrush(QColor("#212121")))
            label_item.setZValue(12)
            label_item.setVisible(False)
            label_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            self.map_scene.addItem(label_item)
            self.map_robot_label = label_item
        else:
            self.map_robot_label = None
        self.map_view.installEventFilter(self)

    def _create_robot_graphics(
        self,
    ) -> tuple[QGraphicsPixmapItem | QGraphicsEllipseItem, QGraphicsLineItem | None]:
        pixmap = self._render_robot_svg()
        if pixmap is not None and not pixmap.isNull():
            item = QGraphicsPixmapItem(pixmap)
            item.setTransformationMode(
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            return item, None
        radius = max(DEFAULT_ROBOT_ICON_SIZE[0], DEFAULT_ROBOT_ICON_SIZE[1]) // 2
        if radius <= 0:
            radius = 10
        circle_pixmap = QPixmap(radius * 2, radius * 2)
        circle_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QPainter(circle_pixmap)
        pen = QPen(QColor("#ff4649"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor("#ff4649")))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.drawEllipse(0, 0, radius * 2, radius * 2)
        painter.end()
        item = QGraphicsPixmapItem(circle_pixmap)
        item.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation)
        return item, None

    def _render_robot_svg(self) -> QPixmap | None:
        if not VECTOR_ICON_PATH.exists():
            return None
        renderer = QSvgRenderer(str(VECTOR_ICON_PATH))
        if not renderer.isValid():
            return None
        desired_width, desired_height = DEFAULT_ROBOT_ICON_SIZE
        target_size = QtCore.QSize(desired_width, desired_height)
        default_size = renderer.defaultSize()
        if not (
            default_size.isValid()
            and default_size.width() > 0
            and default_size.height() > 0
        ):
            default_size = target_size
        image = QImage(default_size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return pixmap
        return pixmap.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

    def _load_map_pixmap(self) -> tuple[QPixmap, float, tuple[float, float, float]]:
        if MAP_IMAGE_PATH:
            pixmap = QPixmap(MAP_IMAGE_PATH)
            if not pixmap.isNull():
                resolution = (
                    MAP_RESOLUTION_OVERRIDE if MAP_RESOLUTION_OVERRIDE > 0 else 0.05
                )
                origin_values = (
                    MAP_ORIGIN_OVERRIDE if MAP_ORIGIN_OVERRIDE else (0.0, 0.0, 0.0)
                )
                origin_x = origin_values[0] if len(origin_values) > 0 else 0.0
                origin_y = origin_values[1] if len(origin_values) > 1 else 0.0
                origin_theta = origin_values[2] if len(origin_values) > 2 else 0.0
                return pixmap, resolution, (origin_x, origin_y, origin_theta)
            print(f"[Map] 지정된 MAP_IMAGE를 불러오지 못했습니다: {MAP_IMAGE_PATH}")
        config_path = MAP_CONFIG_PATH
        try:
            with config_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            image_name = data.get("image")
            resolution = float(data.get("resolution", 0.05))
            origin_raw = data.get("origin") or [0.0, 0.0, 0.0]
            origin_x = float(origin_raw[0]) if len(origin_raw) > 0 else 0.0
            origin_y = float(origin_raw[1]) if len(origin_raw) > 1 else 0.0
            origin_theta = float(origin_raw[2]) if len(origin_raw) > 2 else 0.0
            candidate_paths: list[Path] = []
            if DEFAULT_IMAGE_FALLBACK.exists():
                candidate_paths.append(DEFAULT_IMAGE_FALLBACK.resolve())
            if image_name:
                candidate_paths.append((config_path.parent / image_name).resolve())
            for candidate in candidate_paths:
                pixmap = QPixmap(str(candidate))
                if not pixmap.isNull():
                    return pixmap, resolution, (origin_x, origin_y, origin_theta)
            last_candidate = (
                candidate_paths[-1] if candidate_paths else config_path.parent
            )
            raise FileNotFoundError(f"map image not found: {last_candidate}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[Map] 지도 정보를 불러오지 못했습니다: {exc}")
        if DEFAULT_IMAGE_FALLBACK.exists():
            pixmap = QPixmap(str(DEFAULT_IMAGE_FALLBACK))
            if not pixmap.isNull():
                return pixmap, 0.05, (0.0, 0.0, 0.0)
        fallback = QPixmap(640, 480)
        fallback.fill(QColor("#f5f5f5"))
        painter = QPainter(fallback)
        grid_pen = QPen(QColor("#d0d0d0"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        step = 40
        for x in range(0, fallback.width(), step):
            painter.drawLine(x, 0, x, fallback.height())
        for y in range(0, fallback.height(), step):
            painter.drawLine(0, y, fallback.width(), y)
            painter.end()
        return fallback, 0.05, (0.0, 0.0, 0.0)

    def _fit_map_to_view(self) -> None:
        if (
            self.map_view is None
            or self.map_scene is None
            or self.map_pixmap_item is None
        ):
            return
        rect = self.map_pixmap_item.boundingRect()
        if rect.isNull():
            return
        self.map_view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.map_view and event.type() == QtCore.QEvent.Type.Resize:
            QtCore.QTimer.singleShot(0, self._fit_map_to_view)
        return super().eventFilter(obj, event)

    def _update_video_buttons(self, active: str | None) -> None:
        if self.front_view_button is not None:
            self.front_view_button.setChecked(active == "front")
        if self.arm_view_button is not None:
            self.arm_view_button.setChecked(active == "arm")

    def _ensure_arm_camera_view(self) -> None:
        """선택 단계에서는 로봇 팔 카메라가 표시되도록 보장한다."""
        if self.arm_view_button is None:
            return
        if self.current_robot_id is None:
            return
        if self.active_camera_type == "arm":
            return
        self.arm_view_button.setChecked(True)
        self.on_camera_button_clicked("arm")

    def on_camera_button_clicked(self, camera_type: str) -> None:
        if self.current_robot_id is None:
            QMessageBox.information(
                self,
                "영상 요청",
                "로봇 ID가 없어 영상 스트림을 요청할 수 없습니다.\n주문 생성 또는 로봇 연결 상태를 확인해주세요.",
            )
            self._update_video_buttons(None)
            return
        target_page = self.page_front if camera_type == "front" else self.page_arm
        if self.video_stack is not None and target_page is not None:
            self.video_stack.setCurrentWidget(target_page)
        self._start_video_stream(camera_type)

    def _auto_start_front_camera_if_possible(self) -> None:
        if self.active_camera_type is not None:
            return
        if self.current_robot_id is None:
            if not self._auto_camera_warning_displayed:
                QMessageBox.information(
                    self,
                    "영상 자동 시작 안내",
                    "로봇 ID 정보를 확인할 수 없어 전면 카메라 자동 시작을 건너뜁니다.\n주문 생성 후 다시 시도해주세요.",
                )
                self._auto_camera_warning_displayed = True
            return
        if self.front_view_button is None:
            return
        self._auto_camera_warning_displayed = False
        self.on_camera_button_clicked("front")

    def _set_detection_subscription(self, enabled: bool) -> None:
        """UI 흐름에 맞춰 Pickee 비전 토픽 구독 상태를 전환한다."""
        if self.ros_thread is None:
            return
        try:
            self.ros_thread.set_detection_subscription_enabled(enabled)
        except AttributeError:
            pass

    def _enable_pose_tracking(self) -> None:
        if self.pose_tracking_active:
            return
        self.pose_tracking_active = True
        print("[Map] 픽업 모드로 로봇 위치 추적을 시작합니다.")
        self._set_detection_subscription(True)

    def _disable_pose_tracking(self) -> None:
        if not self.pose_tracking_active:
            self._set_detection_subscription(False)
            return
        self.pose_tracking_active = False
        print("[Map] 픽업 모드를 종료하며 로봇 위치 추적을 중지합니다.")
        self._set_detection_subscription(False)

    def _on_detection_received(self, payload: object) -> None:
        """ROS 비전 bbox 데이터를 받아와 화면에 반영한다."""
        if isinstance(payload, dict):
            self._apply_detection_context(payload)
        normalized = self._normalize_detection_products(payload)
        if normalized is None:
            return
        if not normalized:
            self._clear_bbox_overlays()
            self._clear_detection_selection_if_needed()
            return
        self.pending_detection_products = normalized
        self._bbox_overlays_dirty = True
        self._try_render_bbox_overlays()
        self._update_selection_from_detection(normalized)

    def _apply_detection_context(self, payload: dict[str, Any]) -> None:
        """bbox 토픽에서 전달된 로봇/주문 ID를 UI 상태와 동기화한다."""
        order_id_value: int | None = None
        robot_id_value: int | None = None
        try:
            raw_order_id = payload.get("order_id")
            if raw_order_id is not None:
                order_id_value = int(raw_order_id)
        except (TypeError, ValueError):
            order_id_value = None
        try:
            raw_robot_id = payload.get("robot_id")
            if raw_robot_id is not None:
                robot_id_value = int(raw_robot_id)
        except (TypeError, ValueError):
            robot_id_value = None
        if order_id_value is not None and (
            self.current_order_id is None
            or self.current_order_id == -1
            or self.current_order_id == order_id_value
        ):
            self.current_order_id = order_id_value
        if robot_id_value is not None and (
            self.current_robot_id is None
            or self.current_robot_id == -1
            or self.current_robot_id == robot_id_value
        ):
            self.current_robot_id = robot_id_value

    def _clear_detection_selection_if_needed(self) -> None:
        """bbox 기반 선택지가 표시 중일 때 화면을 초기화한다."""
        if self.selection_options_origin != "detection":
            return
        self.selection_options = []
        self.selection_options_origin = None
        self.selection_selected_index = None
        self.populate_selection_buttons([])
        if self.order_select_stack is not None and self.page_moving_view is not None:
            self.order_select_stack.setCurrentWidget(self.page_moving_view)

    def _update_selection_from_detection(
        self, detection_products: list[dict[str, Any]]
    ) -> None:
        """전면 카메라 bbox 정보를 선택 버튼과 동기화한다."""
        mapped_options: list[dict[str, Any]] = []
        for info in detection_products:
            try:
                bbox_value = int(info.get("bbox_number"))
                product_id_value = int(info.get("product_id"))
            except (TypeError, ValueError):
                continue
            if bbox_value <= 0:
                continue
            name_text = str(info.get("name") or info.get("label") or "").strip()
            if not name_text:
                name_text = f"상품 {product_id_value}"
            label_text = str(info.get("label") or "").strip()
            if not label_text:
                label_text = f"{bbox_value}. {name_text}"
            mapped_options.append(
                {
                    "product_id": product_id_value,
                    "bbox_number": bbox_value,
                    "name": name_text,
                    "label": label_text,
                }
            )
        if not mapped_options:
            self._clear_detection_selection_if_needed()
            return
        self.selection_options = mapped_options
        self.selection_options_origin = "detection"
        self.selection_selected_index = None
        if self.select_title_label is not None:
            self.select_title_label.setText("화면에 표시된 상품을 선택해주세요.")
        self.populate_selection_buttons(mapped_options)
        product_summaries = []
        for opt in mapped_options:
            name_value = opt.get("name")
            bbox_value = opt.get("bbox_number")
            if not name_value or bbox_value is None:
                continue
            product_summaries.append(f"{bbox_value}.{name_value}")
        detection_summary = ", ".join(product_summaries[:3])
        if len(product_summaries) > 3:
            detection_summary += " 외"
        status_label = getattr(self.ui, "label_12", None)
        status_text = "감지된 상품을 선택해주세요."
        if detection_summary:
            status_text = f"감지된 상품을 선택해주세요. ({detection_summary})"
        if status_label is not None:
            status_label.setText(status_text)
        self._update_footer_label(status_text, context="vision_detection_update")
        if self.order_select_stack is not None and self.page_select_product is not None:
            self.order_select_stack.setCurrentWidget(self.page_select_product)

    def _on_bbox_rect_clicked(self, bbox_number: int) -> None:
        """영상 bbox를 클릭했을 때 선택 버튼과 동기화한다."""
        if not self.selection_options:
            return
        target_index: int | None = None
        for index, option in enumerate(self.selection_options):
            try:
                option_bbox = int(option.get("bbox_number"))
            except (TypeError, ValueError):
                continue
            if option_bbox == bbox_number:
                target_index = index
                break
        if target_index is None:
            return
        self.selection_selected_index = target_index
        button: QPushButton | None = None
        if 0 <= target_index < len(self.selection_buttons):
            button = self.selection_buttons[target_index]
        if button is not None:
            if not button.isChecked():
                button.setChecked(True)
                return
        self._refresh_bbox_highlight()

    def _normalize_detection_products(
        self, payload: object
    ) -> list[dict[str, Any]] | None:
        """수신 페이로드를 UI에서 쓰기 쉬운 리스트로 변환한다."""
        if not isinstance(payload, dict):
            return None
        raw_products = payload.get("products")
        if not isinstance(raw_products, list):
            return []
        normalized: list[dict[str, Any]] = []
        for entry in raw_products:
            if not isinstance(entry, dict):
                continue
            bbox_info = entry.get("bbox")
            if not isinstance(bbox_info, dict):
                continue
            coords = self._sanitize_bbox_coords(bbox_info)
            if coords is None:
                continue
            try:
                bbox_number = int(entry.get("bbox_number"))
            except (TypeError, ValueError):
                continue
            if bbox_number <= 0:
                continue
            try:
                product_id = int(entry.get("product_id"))
            except (TypeError, ValueError):
                product_id = -1
            product_name = entry.get("product_name") or entry.get("name")
            if product_name is None:
                product_name = f"상품 {product_id}" if product_id >= 0 else "상품"
            name_text = str(product_name)
            label_text = f"{bbox_number}. {name_text}"
            x1, y1, x2, y2 = coords
            normalized.append(
                {
                    "bbox_number": bbox_number,
                    "product_id": product_id,
                    "name": name_text,
                    "label": label_text,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        return normalized

    def _sanitize_bbox_coords(
        self, bbox: dict[str, Any]
    ) -> tuple[int, int, int, int] | None:
        """UI 좌표계를 벗어난 bbox 좌표를 보정한다."""
        try:
            x1 = int(bbox.get("x1"))
            y1 = int(bbox.get("y1"))
            x2 = int(bbox.get("x2"))
            y2 = int(bbox.get("y2"))
        except (TypeError, ValueError):
            return None
        frame_w = self.VIDEO_FRAME_WIDTH
        frame_h = self.VIDEO_FRAME_HEIGHT
        x1 = max(0, min(x1, frame_w))
        y1 = max(0, min(y1, frame_h))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _try_render_bbox_overlays(self) -> None:
        """팔 카메라 장면 위에 bbox를 그린다."""
        if not self._bbox_overlays_dirty:
            return
        target_scene = self.arm_scene
        target_item = self.arm_item
        if target_scene is None or target_item is None:
            return
        if not self.pending_detection_products:
            self._remove_bbox_items_from_scene()
            self._bbox_overlays_dirty = False
            return
        self._remove_bbox_items_from_scene()
        self.bbox_rect_items.clear()
        self.bbox_label_items.clear()
        self.bbox_label_bg_items.clear()
        pen = QPen(QColor("#ff9800"))
        pen.setWidth(2)
        pen.setCosmetic(True)
        transparent_brush = QBrush(QtCore.Qt.BrushStyle.NoBrush)
        for info in self.pending_detection_products:
            width = info["x2"] - info["x1"]
            height = info["y2"] - info["y1"]
            if width <= 0 or height <= 0:
                continue
            try:
                bbox_number = int(info.get("bbox_number"))
            except (TypeError, ValueError):
                bbox_number = -1
            rect_item = BBoxGraphicsRectItem(
                info["x1"],
                info["y1"],
                width,
                height,
                bbox_number=bbox_number,
                on_click=self._on_bbox_rect_clicked,
                parent=target_item,
            )
            rect_item.setPen(pen)
            rect_item.setBrush(transparent_brush)
            rect_item.setZValue(5)
            self.bbox_overlay_items.append(rect_item)
            if bbox_number > 0:
                self.bbox_rect_items[bbox_number] = rect_item
            label_item = QGraphicsSimpleTextItem(info["label"], target_item)
            label_font = label_item.font()
            if label_font.pointSizeF() < 11.0:
                label_font.setPointSizeF(11.0)
            label_item.setFont(label_font)
            label_item.setBrush(QBrush(QColor("#ffffff")))
            label_item.setZValue(7)
            label_rect = label_item.boundingRect()
            padding = 6.0
            label_x = float(info["x1"])
            max_x = float(self.VIDEO_FRAME_WIDTH) - label_rect.width()
            label_x = max(0.0, min(label_x, max_x))
            label_y = float(info["y1"]) - label_rect.height() - padding
            label_y = max(0.0, label_y)
            label_item.setPos(label_x, label_y)
            bg_item = QGraphicsRectItem(
                label_x - (padding / 2),
                label_y - (padding / 2),
                label_rect.width() + padding,
                label_rect.height() + padding,
                target_item,
            )
            bg_item.setBrush(QBrush(QColor(0, 0, 0, 180)))
            bg_item.setPen(QPen(QtCore.Qt.PenStyle.NoPen))
            bg_item.setZValue(6)
            self.bbox_overlay_items.extend([bg_item, label_item])
            if bbox_number > 0:
                self.bbox_label_items[bbox_number] = label_item
                self.bbox_label_bg_items[bbox_number] = bg_item
        target_scene.update()
        self._bbox_overlays_dirty = False
        self._refresh_bbox_highlight()

    def _remove_bbox_items_from_scene(self) -> None:
        """현재 장면에 올라간 bbox 그래픽을 제거한다."""
        if not self.bbox_overlay_items:
            return
        for item in self.bbox_overlay_items:
            if item is None:
                continue
            scene = item.scene()
            try:
                if scene is not None:
                    scene.removeItem(item)
            except RuntimeError:
                pass
        self.bbox_overlay_items.clear()
        self.bbox_rect_items.clear()
        self.bbox_label_items.clear()
        self.bbox_label_bg_items.clear()

    def _clear_bbox_overlays(self) -> None:
        """bbox 그래픽과 대기 데이터를 모두 초기화한다."""
        self.pending_detection_products = []
        self._bbox_overlays_dirty = False
        self._remove_bbox_items_from_scene()
        self._refresh_bbox_highlight()

    def _refresh_bbox_highlight(self) -> None:
        """선택된 bbox와 동일한 사각형에 강조 색상을 적용한다."""
        if not self.bbox_rect_items:
            return
        selected_bbox = None
        if (
            self.selection_options
            and self.selection_selected_index is not None
            and 0 <= self.selection_selected_index < len(self.selection_options)
        ):
            try:
                selected_bbox = int(
                    self.selection_options[self.selection_selected_index].get(
                        "bbox_number"
                    )
                )
            except (TypeError, ValueError):
                selected_bbox = None
        default_color = QColor("#ff9800")
        highlight_color = QColor("#4caf50")
        default_label_color = QColor("#ffffff")
        default_bg_color = QColor(0, 0, 0, 180)
        highlight_label_color = QColor("#000000")
        highlight_bg_color = QColor("#ffffff")
        for bbox_number, rect_item in self.bbox_rect_items.items():
            if rect_item is None:
                continue
            target_color = (
                highlight_color if selected_bbox == bbox_number else default_color
            )
            pen = QPen(rect_item.pen())
            if pen.color() != target_color:
                pen.setColor(target_color)
                rect_item.setPen(pen)
            label_item = self.bbox_label_items.get(bbox_number)
            if label_item is not None:
                label_color = (
                    highlight_label_color
                    if selected_bbox == bbox_number
                    else default_label_color
                )
                label_item.setBrush(QBrush(label_color))
            bg_item = self.bbox_label_bg_items.get(bbox_number)
            if bg_item is not None:
                bg_color = (
                    highlight_bg_color
                    if selected_bbox == bbox_number
                    else default_bg_color
                )
                bg_item.setBrush(QBrush(bg_color))

    def _on_pickee_status_received(self, msg: object) -> None:
        if not self.pose_tracking_active:
            return
        robot_id_raw = getattr(msg, "robot_id", None)
        try:
            robot_id_value = int(robot_id_raw) if robot_id_raw is not None else None
        except (TypeError, ValueError):
            robot_id_value = None
        if (
            self.current_robot_id is not None
            and robot_id_value is not None
            and robot_id_value != self.current_robot_id
        ):
            return
        x_raw = getattr(msg, "position_x", None)
        y_raw = getattr(msg, "position_y", None)
        if x_raw is None or y_raw is None:
            print("[Map] 위치 좌표가 없는 로봇 상태를 수신했습니다.")
            return
        try:
            x = float(x_raw)
            y = float(y_raw)
        except (TypeError, ValueError):
            print(f"[Map] 유효하지 않은 좌표를 수신했습니다: {x_raw}, {y_raw}")
            return
        theta_raw = getattr(msg, "orientation_z", 0.0)
        try:
            theta = float(theta_raw) if theta_raw is not None else 0.0
        except (TypeError, ValueError):
            theta = 0.0
        if self.current_robot_id is None and robot_id_value is not None:
            self.current_robot_id = robot_id_value
            self._auto_start_front_camera_if_possible()
        print(
            f"[Map] 로봇 상태 수신: robot_id={robot_id_value}, x={x}, y={y}, theta={theta}"
        )
        self.update_robot_pose(x, y, theta)

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        if self.map_pixmap_item is None:
            return 0.0, 0.0
        pixmap = self.map_pixmap_item.pixmap()
        if pixmap.isNull():
            return 0.0, 0.0
        img_w = pixmap.width()
        img_h = pixmap.height()
        if img_w <= 0 or img_h <= 0:
            return 0.0, 0.0
        adjusted_x = y + self.ROBOT_POSITION_OFFSET_Y
        adjusted_y = x + self.ROBOT_POSITION_OFFSET_X
        # x: 왼쪽(최대) → 오른쪽(최소)로 갈수록 감소하므로 반전
        tx = (self.WORLD_X_MAX - adjusted_x) / (self.WORLD_X_MAX - self.WORLD_X_MIN)
        # y: 위(최대) → 아래(최소)로 갈수록 감소하므로 반전
        ty = (self.WORLD_Y_MAX - adjusted_y) / (self.WORLD_Y_MAX - self.WORLD_Y_MIN)
        px = tx * img_w
        py = ty * img_h
        return px, py

    def update_robot_pose(self, x: float, y: float, theta: float) -> None:
        if self.map_robot_item is None:
            return
        px, py = self.world_to_pixel(x, y)
        if not (math.isfinite(px) and math.isfinite(py)):
            return
        # 아이콘 크기를 고려해 중심 정렬
        if isinstance(self.map_robot_item, QGraphicsPixmapItem):
            icon_pixmap = self.map_robot_item.pixmap()
            iw = icon_pixmap.width() if not icon_pixmap.isNull() else 0
            ih = icon_pixmap.height() if not icon_pixmap.isNull() else 0
        else:
            rect = self.map_robot_item.boundingRect()
            iw = rect.width()
            ih = rect.height()
        target_x = px - (iw / 2)
        target_y = py - (ih / 2)
        if iw <= 0 or ih <= 0:
            return
        self.map_robot_item.setVisible(True)
        self.map_robot_item.setTransformOriginPoint(iw / 2, ih / 2)
        self.map_robot_item.setPos(target_x, target_y)
        angle_deg = -math.degrees(theta) + self.ROBOT_ICON_ROTATION_OFFSET_DEG
        self.map_robot_item.setRotation(angle_deg)
        if self.map_robot_label is not None and self.SHOW_ROBOT_LABEL:
            self.map_robot_label.setVisible(True)
            self.map_robot_label.setText(f"({x:.2f}, {y:.2f})")
            label_rect = self.map_robot_label.boundingRect()
            label_x = target_x + (iw / 2) - (label_rect.width() / 2)
            label_y = target_y + ih + self.ROBOT_LABEL_OFFSET_Y
            self.map_robot_label.setPos(label_x, label_y)
        if self.map_scene is not None:
            self.map_scene.update()

    def on_selection_button_toggled(self, button: QPushButton, checked: bool) -> None:
        """선택지 토글 상태를 관리한다."""
        if not checked:
            return
        option_index = button.property("option_index")
        try:
            self.selection_selected_index = int(option_index)
        except (TypeError, ValueError):
            self.selection_selected_index = None
        self._refresh_bbox_highlight()

    def on_select_done_clicked(self) -> None:
        """선택 완료 버튼 클릭을 처리한다."""
        self._submit_current_selection(self.select_done_button)

    def _submit_current_selection(self, trigger_button: QPushButton | None) -> None:
        """선택된 상품을 Main Service에 전달한다."""
        if not self.selection_options:
            QMessageBox.information(self, "상품 선택", "선택 가능한 상품이 없습니다.")
            return
        if self.selection_selected_index is None or not (
            0 <= self.selection_selected_index < len(self.selection_options)
        ):
            QMessageBox.warning(self, "상품 선택", "먼저 선택지를 선택해주세요.")
            return
        if self.current_order_id is None or self.current_robot_id is None:
            QMessageBox.warning(self, "상품 선택", "주문 정보가 확인되지 않습니다.")
            return
        selected = self.selection_options[self.selection_selected_index]
        bbox_number = selected.get("bbox_number")
        product_id = selected.get("product_id")
        try:
            order_id_value = int(self.current_order_id)
            robot_id_value = int(self.current_robot_id)
            bbox_value = int(bbox_number)
            product_id_value = int(product_id)
        except (TypeError, ValueError):
            QMessageBox.warning(
                self, "상품 선택", "선택한 상품 정보를 확인할 수 없습니다."
            )
            return

        self._clear_bbox_overlays()
        if trigger_button is not None:
            trigger_button.setEnabled(False)

        success, error_message = self._perform_product_selection(
            order_id_value,
            robot_id_value,
            bbox_value,
            product_id_value,
        )
        if trigger_button is not None:
            trigger_button.setEnabled(True)

        if not success:
            QMessageBox.warning(
                self, "상품 선택", error_message or "상품 선택을 처리하지 못했습니다."
            )
            return

        QMessageBox.information(self, "상품 선택", "로봇에게 상품 선택을 전달했습니다.")
        return

    def on_auto_pick_clicked(self) -> None:
        """자동 선택 버튼을 눌렀을 때 랜덤 선택을 요청한다."""
        if not self.selection_options:
            QMessageBox.information(self, "자동 선택", "선택 가능한 상품이 없습니다.")
            return
        random_index = random.randrange(len(self.selection_options))
        self.selection_selected_index = random_index
        if 0 <= random_index < len(self.selection_buttons):
            self.selection_buttons[random_index].setChecked(True)
        self._submit_current_selection(self.auto_pick_button)

    def _perform_product_selection(
        self,
        order_id_value: int,
        robot_id_value: int,
        bbox_value: int,
        product_id_value: int,
    ) -> tuple[bool, str]:
        """product_selection 요청을 전송하고 UI 상태를 동기화한다."""
        if self.service_client is None:
            return False, "서비스 클라이언트가 초기화되지 않았습니다."
        self.logger.info(
            "product_selection 요청 시작",
            extra={
                "order_id": order_id_value,
                "robot_id": robot_id_value,
                "bbox_number": bbox_value,
                "product_id": product_id_value,
            },
        )
        try:
            response = self.service_client.select_product(
                order_id=order_id_value,
                robot_id=robot_id_value,
                bbox_number=bbox_value,
                product_id=product_id_value,
            )
        except MainServiceClientError as exc:
            self.logger.warning(
                "product_selection 요청 실패",
                extra={"order_id": order_id_value, "error": str(exc)},
            )
            return False, f"상품을 선택하지 못했습니다.\n{exc}"
        if not response or not response.get("result"):
            if isinstance(response, dict):
                message = response.get("message")
            else:
                message = None
            self.logger.warning(
                "product_selection 응답 실패",
                extra={
                    "order_id": order_id_value,
                    "message": message or "",
                    "response": response,
                },
            )
            return False, message or "상품 선택을 처리하지 못했습니다."
        self.logger.info(
            "product_selection 요청 성공",
            extra={
                "order_id": order_id_value,
                "robot_id": robot_id_value,
                "bbox_number": bbox_value,
                "product_id": product_id_value,
            },
        )
        self._set_selection_status(product_id_value, "선택 진행중")
        self.selection_options = []
        self.selection_options_origin = None
        self.selection_selected_index = None
        self.populate_selection_buttons([])
        self._announce_product_selection_result(product_id_value)
        if self.order_select_stack is not None and self.page_moving_view is not None:
            self.order_select_stack.setCurrentWidget(self.page_moving_view)
        return True, ""

    def _update_selection_voice_button(self, active: bool) -> None:
        if self.selection_voice_button is None:
            return
        text = "on" if active else "off"
        self.selection_voice_button.setChecked(active)
        self.selection_voice_button.setText(text)

    def _handle_selection_voice_result(self, recognized: str) -> bool:
        """음성 인식 결과를 Main Service에 전달해 상품을 선택한다."""
        speech = recognized.strip()
        if not speech:
            return False
        if self.current_order_id is None or self.current_robot_id is None:
            self._show_stt_status_dialog(
                "주문 정보가 없어 음성 선택을 진행할 수 없습니다.",
                icon=QMessageBox.Icon.Warning,
            )
            self._schedule_stt_status_close(2500)
            self._stt_status_hide_timer.start(2500)
            return False
        if self.service_client is None:
            self._show_stt_status_dialog(
                "서비스 클라이언트가 준비되지 않았습니다.",
                icon=QMessageBox.Icon.Warning,
            )
            self._schedule_stt_status_close(2500)
            self._stt_status_hide_timer.start(2500)
            return False
        try:
            order_id_value = int(self.current_order_id)
            robot_id_value = int(self.current_robot_id)
        except (TypeError, ValueError):
            self._show_stt_status_dialog(
                "주문 또는 로봇 정보를 확인할 수 없습니다.",
                icon=QMessageBox.Icon.Warning,
            )
            self._schedule_stt_status_close(2500)
            self._stt_status_hide_timer.start(2500)
            return False

        try:
            response = self.service_client.select_product_by_text(
                order_id=order_id_value,
                robot_id=robot_id_value,
                speech=speech,
            )
        except MainServiceClientError as exc:
            self._show_stt_status_dialog(
                f"음성 선택 요청에 실패했습니다.\n{exc}",
                icon=QMessageBox.Icon.Warning,
            )
            self._schedule_stt_status_close(2500)
            self._stt_status_hide_timer.start(2500)
            return False

        if not response or not response.get("result"):
            message = response.get("message") if isinstance(response, dict) else None
            self._show_stt_status_dialog(
                message or f"'{speech}'에서 선택할 상품을 찾지 못했습니다.",
                icon=QMessageBox.Icon.Warning,
            )
            self._schedule_stt_status_close(2500)
            self._stt_status_hide_timer.start(2500)
            return False

        data = response.get("data") or {}
        bbox_number = data.get("bbox")
        product_id = data.get("product_id")
        try:
            bbox_value = int(bbox_number)
            product_id_value = int(product_id)
        except (TypeError, ValueError):
            self._show_stt_status_dialog(
                "서버 응답에 유효한 bbox 또는 상품 ID가 없습니다.",
                icon=QMessageBox.Icon.Warning,
            )
            self._schedule_stt_status_close(2500)
            self._stt_status_hide_timer.start(2500)
            return False

        self._clear_bbox_overlays()
        success, error_message = self._perform_product_selection(
            order_id_value,
            robot_id_value,
            bbox_value,
            product_id_value,
        )
        if not success:
            self._show_stt_status_dialog(
                error_message or "음성 선택을 처리하지 못했습니다.",
                icon=QMessageBox.Icon.Warning,
            )
            self._schedule_stt_status_close(2500)
            self._stt_status_hide_timer.start(2500)
            return False

        self._show_stt_status_dialog(
            f"'{speech}' 음성으로 선택을 전달했습니다.",
            icon=QMessageBox.Icon.Information,
        )
        self._schedule_stt_status_close(2000)
        self._stt_status_hide_timer.start(2000)
        return True

    def _start_video_stream(self, camera_type: str) -> None:
        if self.current_robot_id is None:
            return
        if self.active_camera_type == camera_type and self.video_receiver is not None:
            if self.video_receiver.isRunning():
                self._update_video_buttons(camera_type)
                return
        # 카메라 전환 시에는 서버 스트림 유지가 필요하므로 stop 요청을 보내지 않는다.
        self._stop_video_stream(send_request=False)
        user_type = self._current_user_type()
        try:
            response = self.service_client.start_video_stream(
                user_id=self.current_user_id,
                user_type=user_type,
                robot_id=int(self.current_robot_id),
                camera_type=camera_type,
            )
        except MainServiceClientError as exc:
            QMessageBox.warning(self, "영상 요청 실패", str(exc))
            self._update_video_buttons(None)
            return
        if not response or not response.get("result", False):
            message = response.get("message") if isinstance(response, dict) else None
            QMessageBox.warning(
                self,
                "영상 요청 실패",
                message or "영상 스트림을 시작할 수 없습니다.",
            )
            self._update_video_buttons(None)
            return
        self.active_camera_type = camera_type
        self.video_receiver = VideoStreamReceiver(
            robot_id=int(self.current_robot_id),
            camera_type=camera_type,
            parent=self,
        )
        self.video_receiver.frame_received.connect(self.on_video_frame_received)
        self.video_receiver.error_occurred.connect(self.on_video_stream_error)
        self.video_receiver.start()
        self._update_video_buttons(camera_type)

    def _stop_video_stream(self, *, send_request: bool) -> None:
        """비디오 스트림을 중지하고 관련 리소스를 정리합니다.

        Args:
            send_request: 서버에 스트리밍 중지 요청을 보낼지 여부
        """
        # 비디오 수신기 정리
        if self.video_receiver is not None:
            try:
                self.video_receiver.stop()
                self.video_receiver = None
                self.logger.debug("비디오 수신기 정지 완료")
            except Exception as e:
                self.logger.warning(f"비디오 수신기 정지 중 오류: {e}")

        # 서버에 스트리밍 중지 요청
        if (
            send_request
            and self.active_camera_type is not None
            and self.current_robot_id is not None
            and self.service_client is not None
        ):
            user_type = self._current_user_type()
            try:
                self.logger.debug("서버에 비디오 스트림 중지 요청")
                self.service_client.stop_video_stream(
                    user_id=self.current_user_id,
                    user_type=user_type,
                    robot_id=int(self.current_robot_id),
                )
                self.logger.debug("비디오 스트림 중지 요청 성공")
            except MainServiceClientError as e:
                self.logger.error(f"비디오 스트림 중지 요청 실패: {e}")

        # 화면에서 비디오 표시 제거
        if self.front_scene is not None and self.front_item is not None:
            try:
                self.front_scene.removeItem(self.front_item)
                self.front_item = None
            except Exception as e:
                self.logger.warning(f"전면 비디오 화면 정리 오류: {e}")
        self._clear_bbox_overlays()

        if self.arm_scene is not None and self.arm_item is not None:
            try:
                self.arm_scene.removeItem(self.arm_item)
                self.arm_item = None
            except Exception as e:
                self.logger.warning(f"암 카메라 화면 정리 오류: {e}")

        # 상태 초기화
        self.active_camera_type = None
        self._update_video_buttons(None)

    def _current_user_type(self) -> str:
        if isinstance(self.user_info, dict):
            value = self.user_info.get("user_type") or self.user_info.get("role")
            if value:
                return str(value)
        return "user"

    def on_video_frame_received(self, camera_type: str, image: QImage) -> None:
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        scene = self.front_scene if camera_type == "front" else self.arm_scene
        item_attr = "front_item" if camera_type == "front" else "arm_item"
        view = self.gv_front if camera_type == "front" else self.gv_arm
        if scene is None or view is None:
            return
        item = getattr(self, item_attr)
        if item is None:
            item = scene.addPixmap(pixmap)
            setattr(self, item_attr, item)
        else:
            item.setPixmap(pixmap)
        if VIDEO_STREAM_DEBUG:
            print(
                f"[VideoStream] frame camera={camera_type} size={pixmap.width()}x{pixmap.height()}"
            )
        self._fit_view(view, item)
        if camera_type == "arm":
            self._try_render_bbox_overlays()
        scene.update()

    def _fit_view(self, view: QGraphicsView, item) -> None:
        if item is None:
            return
        rect = item.boundingRect()
        if rect.isNull():
            return
        view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def on_video_stream_error(self, message: str) -> None:
        self._stop_video_stream(send_request=False)
        QMessageBox.warning(self, "영상 스트림 오류", message)

    def _build_order_auto_select_map(
        self,
        order_data: dict[str, Any] | None,
    ) -> dict[int, bool]:
        # 서버 응답의 auto_select 값을 기준으로 상품 분류 정보를 구성한다.
        auto_select_map: dict[int, bool] = {}
        if not isinstance(order_data, dict):
            return auto_select_map
        products = order_data.get("products")
        if not isinstance(products, list):
            return auto_select_map
        for entry in products:
            if not isinstance(entry, dict):
                continue
            product_id_value = entry.get("product_id")
            try:
                product_id = int(product_id_value)
            except (TypeError, ValueError):
                continue
            auto_flag = entry.get("auto_select")
            auto_select_map[product_id] = bool(auto_flag)
        return auto_select_map

    def categorize_cart_items(
        self,
        items: list[CartItemData],
        auto_select_map: dict[int, bool] | None = None,
    ) -> tuple[list[CartItemData], list[CartItemData]]:
        # auto_select 플래그를 사용해 자동 및 원격 선택 상품을 구분한다.
        if not items:
            return [], []
        reference_map = auto_select_map or {}
        remote_items: list[CartItemData] = []
        auto_items: list[CartItemData] = []
        for item in items:
            if reference_map.get(item.product_id, False):
                auto_items.append(item)
            else:
                remote_items.append(item)
        return remote_items, auto_items

    def populate_selection_list(
        self,
        list_widget,
        items: list[CartItemData],
        status_text: str,
    ) -> None:
        if list_widget is None:
            return

        list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection,
        )
        list_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        list_widget.setSpacing(6)
        list_widget.clear()
        if not items:
            list_widget.addItem(QListWidgetItem("표시할 상품이 없습니다."))
            return

        for index, item in enumerate(items, start=1):
            widget = CartSelectItemWidget()
            existing_state = self.selection_item_states.get(item.product_id)
            picked_count = int(existing_state.get("picked", 0)) if existing_state else 0
            base_status = (
                str(existing_state.get("base_status"))
                if existing_state and existing_state.get("base_status")
                else status_text
            )
            display_status = "완료" if picked_count >= item.quantity else base_status
            widget.apply_item(
                index=index,
                name=item.name,
                quantity=item.quantity,
                status_text=display_status,
                picked=picked_count,
                image_path=item.image_path,
            )
            list_item = QListWidgetItem()
            list_item.setSizeHint(widget.sizeHint())
            list_widget.addItem(list_item)
            list_widget.setItemWidget(list_item, widget)
            self.selection_item_states[item.product_id] = {
                "widget": widget,
                "total": item.quantity,
                "picked": picked_count,
                "base_status": base_status,
                "name": item.name,
            }
        if status_text.startswith("원격"):
            self._update_selection_state_label()

    def _update_selection_progress(self, product_id: int, picked_delta: int) -> None:
        """선택 리스트에 표시된 상품 진행률을 갱신한다."""
        state = self.selection_item_states.get(product_id)
        if state is None:
            return
        total = int(state.get("total", 0))
        picked = int(state.get("picked", 0)) + picked_delta
        if total <= 0:
            return
        picked = max(0, min(total, picked))
        state["picked"] = picked
        if not state.get("name"):
            state["name"] = self._get_product_name_by_id(product_id)
        widget = state.get("widget")
        if isinstance(widget, CartSelectItemWidget):
            widget.label_progress.setText(f"({picked}/{total})")
            if picked >= total:
                widget.set_status("완료")
            else:
                base_status = str(state.get("base_status") or "대기")
                widget.set_status(base_status)
        self._refresh_selection_progress_summary()
        self._update_selection_state_label(product_id)

    def _show_pick_complete_page(self) -> None:
        if self.order_select_stack is None or self.page_end_pick_up is None:
            return
        self.order_select_stack.setCurrentWidget(self.page_end_pick_up)

    def _set_selection_status(self, product_id: int, status_text: str) -> None:
        """선택 항목의 상태 텍스트를 갱신한다."""
        state = self.selection_item_states.get(product_id)
        if state is None:
            return
        state["base_status"] = status_text
        if not state.get("name"):
            state["name"] = self._get_product_name_by_id(product_id)
        widget = state.get("widget")
        if isinstance(widget, CartSelectItemWidget):
            widget.set_status(status_text)
        self._update_selection_state_label(product_id)

    def _show_moving_page(self) -> None:
        """선택 단계가 아니고 쇼핑이 완료되지 않았다면 이동 화면을 표시한다."""
        if self.pick_flow_completed:
            return
        if self.order_select_stack is None or self.page_moving_view is None:
            return
        current_widget = self.order_select_stack.currentWidget()
        if current_widget is self.page_select_product:
            return
        self.order_select_stack.setCurrentWidget(self.page_moving_view)

    def _update_footer_label(
        self,
        text: str,
        *,
        context: str | None = None,
        force: bool = False,
    ) -> None:
        """푸터 상태 문구를 갱신하고 로깅한다."""
        if self._footer_locked and not force:
            return
        footer_label = getattr(self.ui, "label_robot_notification", None)
        if footer_label is not None:
            footer_label.setText(text)
        context_prefix = f"[{context}] " if context else ""
        self.logger.info(f"{context_prefix}footer_label: {text}")

    def _announce_product_selection_result(self, product_id: int) -> None:
        """상품 선택 성공 시 사용자에게 문구를 노출한다."""
        product_name = self._get_product_name_by_id(product_id) or "상품"
        message = f"{product_name}을 선택했습니다."
        status_label = getattr(self.ui, "label_12", None)
        if not self.pick_flow_completed and status_label is not None:
            status_label.setText(message)
        self._update_footer_label(message, context="product_selection_response")

    def _get_friendly_section_name(self, section_label: str | None) -> str | None:
        """섹션 식별자를 사용자 친화적인 매대 이름으로 변환한다."""
        if not section_label:
            return None
        normalized = str(section_label).strip().upper()
        return SECTION_FRIENDLY_NAMES.get(normalized)

    def _mark_all_selection_completed(self) -> None:
        """선택 항목 전체를 완료 상태로 동기화한다."""
        if not self.selection_item_states:
            return
        for state in self.selection_item_states.values():
            total = int(state.get("total", 0))
            state["picked"] = total
            state["base_status"] = "완료"
            widget = state.get("widget")
            if isinstance(widget, CartSelectItemWidget):
                widget.label_progress.setText(f"({total}/{total})")
                widget.set_status("완료")
        self._refresh_selection_progress_summary()
        self._update_selection_state_label()

    def _refresh_selection_progress_summary(self) -> None:
        """전체 진행률을 합산해 Progress Bar와 텍스트에 반영한다."""
        if not self.selection_item_states:
            if self.progress_bar is not None:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
            if self.progress_text is not None:
                self.progress_text.setText("0%")
            return
        total_required = 0
        total_picked = 0
        for state in self.selection_item_states.values():
            total_required += max(0, int(state.get("total", 0)))
            total_picked += max(0, int(state.get("picked", 0)))
        total_required = max(total_required, 0)
        total_picked = min(total_required, total_picked)
        progress_ratio = 0
        if total_required > 0:
            progress_ratio = round((total_picked / total_required) * 100)
        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(progress_ratio)
        if self.progress_text is not None:
            self.progress_text.setText(f"{progress_ratio}%")

    def _update_selection_state_label(self, product_id: int | None = None) -> None:
        if self.selection_state_label is None:
            return
        display_text = "선택 대기 중"
        if not hasattr(self, "selection_item_states"):
            self.selection_state_label.setText(display_text)
            return
        remote_items = getattr(self, "remote_selection_items", [])
        auto_items = getattr(self, "auto_selection_items", [])
        active_state = None
        active_product_id = product_id
        if active_product_id is not None:
            active_state = self.selection_item_states.get(active_product_id)
        if active_state is None:
            for pid, info in self.selection_item_states.items():
                base_status = str(info.get("base_status") or "")
                if base_status.replace(" ", "") == "선택진행중":
                    active_state = info
                    active_product_id = pid
                    break
        if active_state is None:
            for item in remote_items:
                info = self.selection_item_states.get(item.product_id)
                if info is None:
                    continue
                total = max(0, int(info.get("total", 0)))
                picked = max(0, int(info.get("picked", 0)))
                if total <= 0:
                    continue
                if picked < total:
                    active_state = info
                    active_product_id = item.product_id
                    break
        if active_state is None:
            for pid, info in self.selection_item_states.items():
                total = max(0, int(info.get("total", 0)))
                picked = max(0, int(info.get("picked", 0)))
                if total <= 0 or picked >= total:
                    continue
                active_state = info
                active_product_id = pid
                break
        if active_state is None:
            self.selection_state_label.setText(display_text)
            return
        total = max(0, int(active_state.get("total", 0)))
        picked = max(0, int(active_state.get("picked", 0)))
        name = active_state.get("name")
        if not name and active_product_id is not None:
            name = self._get_product_name_by_id(active_product_id)
        if not name:
            name = "상품"
        if total > 0:
            if picked < total:
                current_index = picked + 1
            else:
                current_index = total
            progress_text = f" ({current_index}/{total})"
        else:
            progress_text = ""
        self.selection_state_label.setText(f"{name} 선택중{progress_text}")

    def _get_product_name_by_id(self, product_id: int) -> str | None:
        remote_items = getattr(self, "remote_selection_items", [])
        auto_items = getattr(self, "auto_selection_items", [])
        for item in remote_items:
            if item.product_id == product_id:
                return item.name
        for item in auto_items:
            if item.product_id == product_id:
                return item.name
        return None

    def open_profile_dialog(self) -> None:
        dialog = getattr(self, "profile_dialog", None)
        button = getattr(self.ui, "btn_profile", None)
        if dialog is None or button is None:
            return

        dialog.set_user_info(self.user_info)
        dialog.adjustSize()
        anchor = button.mapToGlobal(button.rect().bottomRight())
        x = anchor.x() - dialog.width()
        y = anchor.y() + 6

        parent_widget = self.window()
        if parent_widget is not None:
            parent_top_left = parent_widget.mapToGlobal(QtCore.QPoint(0, 0))
            parent_rect = QtCore.QRect(parent_top_left, parent_widget.size())
            if x < parent_rect.left():
                x = parent_rect.left()
            if x + dialog.width() > parent_rect.right():
                x = parent_rect.right() - dialog.width()
            if y + dialog.height() > parent_rect.bottom():
                y = anchor.y() - dialog.height() - 6
            if y < parent_rect.top():
                y = parent_rect.top()
        else:
            window_handle = button.window().windowHandle() if button.window() else None
            screen = window_handle.screen() if window_handle is not None else None
            if screen is not None:
                available = screen.availableGeometry()
                if x < available.left():
                    x = available.left()
                if x + dialog.width() > available.right():
                    x = available.right() - dialog.width()
                if y + dialog.height() > available.bottom():
                    y = anchor.y() - dialog.height() - 6
                if y < available.top():
                    y = available.top()

        dialog.move(int(x), int(y))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def on_logout_requested(self) -> None:
        """로그아웃을 요청하고 리소스를 정리한 뒤 창을 닫습니다."""
        self.logger.info("로그아웃 요청 수신")

        # 이미 정리 중인 경우 중복 실행 방지
        if self._cleanup_requested.is_set():
            return

        # 사용자 확인
        reply = QMessageBox.question(
            self,
            "로그아웃",
            "정말 로그아웃 하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            self._cleanup_requested.clear()
            return

        self.logger.info("로그아웃 시작")
        success = True

        try:
            # UI 비활성화
            self.setEnabled(False)
            QtCore.QCoreApplication.processEvents()

            # 비디오 스트림 중지
            self.logger.debug("비디오 스트림 정리 시작")
            self._stop_video_stream(send_request=True)
            self.logger.debug("비디오 스트림 정리 완료")

            # 알림 클라이언트 정리
            if self.notification_client is not None:
                self.logger.debug("알림 클라이언트 정리 시작")
                self.notification_client.stop()
                self.notification_client = None
                self.logger.debug("알림 클라이언트 정리 완료")

            # ROS 노드 연결 해제 및 종료
            if self.ros_thread is not None:
                self.logger.debug("ROS 노드 정리 시작")
                try:
                    # 모든 시그널 연결 해제
                    self.ros_thread.disconnect()
                    # ROS 노드 종료
                    self.ros_thread.shutdown()
                    # 스레드 종료 대기 (최대 5초)
                    if not self.ros_thread.wait(5000):
                        self.logger.warning("ROS 스레드 종료 타임아웃")
                        success = False
                    self.ros_thread = None
                except Exception as e:
                    self.logger.error(f"ROS 노드 정리 중 오류: {e}")
                    success = False
                self.logger.debug("ROS 노드 정리 완료")

            # 음성 인식 리소스 정리
            self.logger.debug("음성 인식 리소스 정리 시작")
            self._shutdown_speech_recognition()
            self.logger.debug("음성 인식 리소스 정리 완료")

            # 위치 추적 중지
            self._disable_pose_tracking()

            # UI 상태 초기화
            self.logger.debug("UI 상태 초기화 시작")
            self._cleanup_cart()
            self._cleanup_selection()
            self._cleanup_ui_states()
            self.logger.debug("UI 상태 초기화 완료")

        except Exception as e:
            self.logger.exception("리소스 정리 중 오류 발생")
            success = False

        finally:
            self._cleanup_requested.clear()
            if success:
                self.logger.info("로그아웃 성공")
            else:
                self.logger.warning("일부 리소스 정리 실패")

            # 결과와 관계없이 창 닫기
            self.closed.emit()
            self.close()

    def _cleanup_cart(self) -> None:
        """장바구니 위젯과 데이터를 정리합니다."""
        try:
            # 장바구니 데이터 초기화
            self.cart_items.clear()
            self.logger.debug("장바구니 데이터 초기화 완료")

            # 장바구니 위젯 정리
            self.cart_widgets.clear()
            if self.cart_items_layout is not None:
                for i in reversed(range(self.cart_items_layout.count())):
                    item = self.cart_items_layout.itemAt(i)
                    if item is not None:
                        spacer = item.spacerItem()
                        if spacer is not None and spacer is self.cart_spacer:
                            continue
                        widget = item.widget()
                        if widget is None or widget is self.cart_empty_label:
                            continue
                        widget.setParent(None)
            self.logger.debug("장바구니 위젯 정리 완료")
            self.update_cart_empty_state()

        except Exception as e:
            self.logger.error(f"장바구니 정리 중 오류: {e}")

    def _cleanup_ui_states(self) -> None:
        """모든 UI 위젯의 상태를 초기화합니다."""
        try:
            # 검색 입력창 초기화
            if self.search_input is not None:
                self.search_input.clear()

            # 알러지 체크박스 초기화
            if self.allergy_total_checkbox is not None:
                self.allergy_total_checkbox.setChecked(False)
            for checkbox in self.allergy_checkbox_map.values():
                checkbox.setChecked(False)
            if self.vegan_checkbox is not None:
                self.vegan_checkbox.setChecked(False)

            # 비디오 표시 영역 초기화
            if self.gv_front is not None:
                self.gv_front.viewport().update()
            if self.gv_arm is not None:
                self.gv_arm.viewport().update()

            # 지도 표시 초기화
            if self.map_view is not None:
                self.map_view.viewport().update()

            # 카메라 버튼 상태 초기화
            if self.front_view_button is not None:
                self.front_view_button.setChecked(False)
            if self.arm_view_button is not None:
                self.arm_view_button.setChecked(False)

            # 모드 상태 초기화
            if self.shopping_button is not None:
                self.shopping_button.setChecked(True)
            if self.store_button is not None:
                self.store_button.setChecked(False)

            # 장바구니 UI 초기화
            if self.cart_toggle_button is not None:
                self.cart_toggle_button.setChecked(False)
            self.cart_expanded = False
            self.apply_cart_state()

            # 상품 목록 초기화
            self.products.clear()
            self.all_products.clear()
            self.product_index.clear()
            self.refresh_product_grid()

            self.logger.debug("UI 상태 초기화 완료")

        except Exception as e:
            self.logger.error(f"UI 상태 초기화 중 오류: {e}")

    def _cleanup_selection(self) -> None:
        """상품 선택 관련 데이터를 정리합니다."""
        try:
            # 선택 버튼 정리
            for button in self.selection_buttons:
                if isinstance(button, QPushButton):
                    self.selection_button_group.removeButton(button)
                    button.setParent(None)
            self.selection_buttons.clear()

            # 선택 데이터 초기화
            self.selection_options.clear()
            self.selection_options_origin = None
            self.selection_selected_index = None
            self.remote_selection_items.clear()
            self.auto_selection_items.clear()
            self.selection_item_states.clear()

            # 선택 버튼 그리드 초기화
            if self.selection_grid is not None:
                self._initialize_selection_grid()
            self._update_selection_state_label()
            if self.selection_state_label is not None:
                self.selection_state_label.setText("선택 대기 중")

            self.logger.debug("상품 선택 UI 정리 완료")

        except Exception as e:
            self.logger.error(f"상품 선택 정리 중 오류: {e}")

    def _disconnect_ros(self) -> None:
        """ROS 연결을 해제합니다."""
        self._disable_pose_tracking()
        if self.ros_thread is not None:
            try:
                self.ros_thread.pickee_status_received.disconnect(
                    self._on_pickee_status_received
                )
            except (TypeError, RuntimeError):
                pass

    def _cleanup_stt(self) -> None:
        """음성 인식 관련 리소스를 정리합니다."""
        if hasattr(self, "_stt_module"):
            self._shutdown_speech_recognition()

    def _cleanup_cart(self) -> None:
        """장바구니 관련 리소스를 정리합니다."""
        if hasattr(self, "cart_items"):
            self.cart_items.clear()
        if hasattr(self, "cart_widgets"):
            for widget in self.cart_widgets.values():
                try:
                    if widget is not None:
                        widget.setParent(None)
                        widget.deleteLater()
                except:
                    pass
            self.cart_widgets.clear()

    def _cleanup_selection(self) -> None:
        """선택 관련 상태를 정리합니다."""
        if hasattr(self, "selection_item_states"):
            for state in self.selection_item_states.values():
                try:
                    widget = state.get("widget")
                    if widget is not None:
                        widget.setParent(None)
                        widget.deleteLater()
                except:
                    pass
            self.selection_item_states.clear()
            self._update_selection_state_label()
        if (
            hasattr(self, "selection_state_label")
            and self.selection_state_label is not None
        ):
            self.selection_state_label.setText("선택 대기 중")

    def _update_user_header(self) -> None:
        name = str(
            self.user_info.get("name") or self.user_info.get("user_id") or "사용자"
        )
        label = getattr(self.ui, "label_user_name", None)
        if label is not None:
            label.setText(f"{name} 님")

    def set_mode(self, mode):
        if mode == self.current_mode:
            return

        self.current_mode = mode

        if mode == "shopping":
            if self.shopping_button:
                self.shopping_button.setChecked(True)
            if self.store_button:
                self.store_button.setChecked(False)
            self._disable_pose_tracking()
            if self.search_input is not None:
                self.search_input.show()
            if self.search_button is not None:
                self.search_button.show()
            if self.mic_button is not None:
                self.mic_button.show()
            self.show_main_page(self.page_user)
            self.show_side_page(self.side_pick_filter_page)
            return

        if mode == "pick":
            if self.store_button:
                self.store_button.setChecked(True)
            if self.shopping_button:
                self.shopping_button.setChecked(False)
            if self.search_input is not None:
                self.search_input.hide()
            if self.search_button is not None:
                self.search_button.hide()
            if self.mic_button is not None:
                self.mic_button.hide()
            self.show_main_page(self.page_pick)
            self.show_side_page(self.side_shop_page)
            self._enable_pose_tracking()

    def show_main_page(self, page):
        if self.main_stack is None or page is None:
            return
        if self.main_stack.currentWidget() is page:
            return
        self.main_stack.setCurrentWidget(page)

    def show_side_page(self, page):
        if self.side_stack is None or page is None:
            return
        if self.side_stack.currentWidget() is page:
            return
        self.side_stack.setCurrentWidget(page)

    def apply_cart_state(self):
        if self.cart_toggle_button is not None:
            tooltip = "장바구니 접기" if self.cart_expanded else "장바구니 펼치기"
            self.cart_toggle_button.setToolTip(tooltip)
            if self.cart_icon_up is not None or self.cart_icon_down is not None:
                icon = (
                    self.cart_icon_down
                    if self.cart_expanded and self.cart_icon_down is not None
                    else self.cart_icon_up
                )
                if icon is not None:
                    self.cart_toggle_button.setIcon(icon)

        if self.cart_body:
            self.cart_body.setVisible(self.cart_expanded)
            if self.cart_expanded:
                self.render_all_cart_items()

        if self.product_scroll is not None:
            self.product_scroll.setVisible(not self.cart_expanded)

        if self.cart_container_layout is not None:
            top_margin = (
                self.cart_margin_expanded_top
                if self.cart_expanded
                else self.cart_margin_collapsed_top
            )
            self.cart_container_layout.setContentsMargins(
                self.cart_margin_left,
                top_margin,
                self.cart_margin_right,
                self.cart_margin_bottom,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_product_grid()

    def refresh_product_grid(self) -> None:
        if self.product_grid is None:
            return

        columns = self.calculate_columns()
        if columns <= 0:
            columns = 1

        if columns == self.current_columns and self.products:
            return

        self.populate_products(self.products, columns)
        self.current_columns = columns

    def calculate_columns(self) -> int:
        scroll_area = getattr(self.ui, "scrollArea", None)
        if scroll_area is not None:
            available_width = scroll_area.viewport().width()
        elif self.products_container is not None:
            available_width = self.products_container.width()
        else:
            available_width = self.width()

        if available_width <= 0:
            return 1

        if self.product_grid is not None:
            margins = self.product_grid.contentsMargins()
            available_width -= margins.left() + margins.right()

        spacing = self.product_grid.horizontalSpacing() if self.product_grid else 0
        if spacing < 0:
            spacing = 0

        card_width = ProductCard.DEFAULT_SIZE.width()
        total_per_card = card_width + spacing
        if total_per_card <= 0:
            return 1

        columns = max(1, (available_width + spacing) // total_per_card)
        if self.products:
            columns = min(columns, len(self.products))
        return int(columns)

    def populate_products(
        self,
        products: list[ProductData],
        columns: int,
    ) -> None:
        if self.product_grid is None:
            return

        while self.product_grid.count():
            item = self.product_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for col in range(columns + 1):
            self.product_grid.setColumnStretch(col, 0)

        if not products:
            message = getattr(
                self,
                "empty_products_message",
                "표시할 상품이 없습니다.",
            )
            placeholder = QLabel(message)
            placeholder.setObjectName("product_grid_placeholder")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setWordWrap(True)
            span = max(1, columns)
            self.product_grid.addWidget(
                placeholder,
                0,
                0,
                1,
                span,
                QtCore.Qt.AlignmentFlag.AlignCenter,
            )
            self.product_grid.setColumnStretch(0, 1)
            self.product_grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            return

        for index, product in enumerate(products):
            row = index // columns
            col = index % columns
            card = ProductCard()
            # 사용자의 알러지 정보와 함께 상품 정보 적용
            user_allergy = None
            if self.user_info is not None and "allergy_info" in self.user_info:
                user_allergy = self.user_info["allergy_info"]
            card.apply_product(product, user_allergy)
            button = getattr(card.ui, "btn_add_product", None)
            if button is None:
                button = getattr(card.ui, "toolButton", None)
            if button is not None:
                button.clicked.connect(lambda _, p=product: self.on_add_to_cart(p))
            self.product_grid.addWidget(card, row, col)

        rows = (len(products) + columns - 1) // columns if products else 0
        spacer = QSpacerItem(
            0,
            0,
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.product_grid.addItem(spacer, 0, columns, max(1, rows), 1)
        self.product_grid.setColumnStretch(columns, 1)
        self.product_grid.setHorizontalSpacing(16)
        self.product_grid.setVerticalSpacing(16)
        self.product_grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.product_grid.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )

    def on_add_to_cart(self, product: ProductData) -> None:
        item = self.cart_items.get(product.product_id)
        if item is not None:
            item.quantity = min(item.quantity + 1, CartItemWidget.MAX_QUANTITY)
        else:
            item = CartItemData(
                product_id=product.product_id,
                name=product.name,
                quantity=1,
                price=product.discounted_price,
                image_path=product.image_path,
            )
            self.cart_items[product.product_id] = item
        self.render_cart_item(product.product_id)
        self.update_cart_summary()
        self.sync_select_all_state()

    def render_all_cart_items(self) -> None:
        if self.cart_items_layout is None:
            return
        for product_id in list(self.cart_widgets.keys()):
            if product_id not in self.cart_items:
                self.remove_cart_widget(product_id)
        for product_id in self.cart_items.keys():
            self.render_cart_item(product_id)
        self.sync_select_all_state()
        self.update_cart_summary()

    def render_cart_item(self, product_id: int) -> None:
        if self.cart_items_layout is None:
            return
        item = self.cart_items.get(product_id)
        if item is None:
            return
        widget = self.cart_widgets.get(product_id)
        if widget is None:
            widget = CartItemWidget()
            widget.quantity_changed.connect(self.on_item_quantity_changed)
            widget.remove_requested.connect(self.on_item_remove_requested)
            widget.checked_changed.connect(self.on_item_checked_changed)
            insert_index = self.cart_items_layout.count()
            if self.cart_spacer is not None:
                insert_index = max(0, insert_index - 1)
            self.cart_items_layout.insertWidget(insert_index, widget)
            self.cart_widgets[product_id] = widget
        widget.apply_item(item)

    def remove_cart_widget(self, product_id: int) -> None:
        widget = self.cart_widgets.pop(product_id, None)
        if widget is not None:
            widget.setParent(None)

    def on_item_remove_requested(self, product_id: int) -> None:
        if product_id in self.cart_items:
            del self.cart_items[product_id]
        self.remove_cart_widget(product_id)
        self.update_cart_summary()
        self.sync_select_all_state()

    def on_item_quantity_changed(self, product_id: int, quantity: int) -> None:
        item = self.cart_items.get(product_id)
        if item is None:
            return
        item.quantity = quantity
        self.render_cart_item(product_id)
        self.update_cart_summary()

    def on_item_checked_changed(self, product_id: int, checked: bool) -> None:
        item = self.cart_items.get(product_id)
        if item is None:
            return
        item.is_selected = checked
        self.update_cart_summary()
        self.sync_select_all_state()

    def on_select_all_changed(self, state: int) -> None:
        if not self.cart_items:
            self.ui.chk_select_all.blockSignals(True)
            self.ui.chk_select_all.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.ui.chk_select_all.blockSignals(False)
            return
        checked = state == QtCore.Qt.CheckState.Checked
        for product_id, item in self.cart_items.items():
            item.is_selected = checked
            widget = self.cart_widgets.get(product_id)
            if widget is None:
                self.render_cart_item(product_id)
                widget = self.cart_widgets.get(product_id)
            if widget is not None:
                widget.set_checked(checked)
        self.update_cart_summary()
        self.sync_select_all_state()

    def on_delete_selected_clicked(self) -> None:
        for product_id, item in list(self.cart_items.items()):
            if item.is_selected:
                self.on_item_remove_requested(product_id)

    def sync_select_all_state(self) -> None:
        if not self.cart_items:
            self.ui.chk_select_all.blockSignals(True)
            self.ui.chk_select_all.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.ui.chk_select_all.blockSignals(False)
            return
        selected = sum(1 for item in self.cart_items.values() if item.is_selected)
        state = (
            QtCore.Qt.CheckState.Checked
            if selected == len(self.cart_items)
            else QtCore.Qt.CheckState.Unchecked
        )
        self.ui.chk_select_all.blockSignals(True)
        self.ui.chk_select_all.setCheckState(state)
        self.ui.chk_select_all.blockSignals(False)

    def update_cart_summary(self) -> None:
        total_qty = sum(
            item.quantity for item in self.cart_items.values() if item.is_selected
        )
        total_price = sum(
            item.total_price for item in self.cart_items.values() if item.is_selected
        )
        label_quantity = getattr(self.ui, "label_total_count", None)
        label_amount = getattr(self.ui, "label_pay_price", None)
        if label_quantity is not None:
            label_quantity.setText(f"{total_qty}")
        if label_amount is not None:
            font = label_amount.font()
            font.setPointSize(15)
            label_amount.setFont(font)
            formatted_text = f'<span style="color: #000000; font-weight: 600;">총액 : </span><span style="font-weight: 800;">{total_price:,}</span><span style="color: #000000; font-weight: 600;">원</span>'
            label_amount.setText(formatted_text)
            # HTML 해석 활성화
            label_amount.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.update_cart_empty_state()

    def update_cart_empty_state(self) -> None:
        if self.cart_empty_label is None:
            return
        is_empty = not self.cart_items
        self.cart_empty_label.setVisible(is_empty)

    def set_products(self, products: list[ProductData]) -> None:
        self.all_products = list(products)
        self._apply_product_filters(refresh=False)
