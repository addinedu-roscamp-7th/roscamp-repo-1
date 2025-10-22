import math
import os
from pathlib import Path
from dataclasses import replace
from typing import Any

from PyQt6 import QtCore
from PyQt6.QtGui import QIcon
from PyQt6.QtGui import QPixmap
from PyQt6.QtGui import QTransform
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView
from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QSpacerItem
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QWidget

from shopee_app.services.app_notification_client import AppNotificationClient
from shopee_app.services.main_service_client import MainServiceClient
from shopee_app.services.main_service_client import MainServiceClientError
from shopee_app.pages.models.cart_item_data import CartItemData
from shopee_app.pages.models.product_data import ProductData
from shopee_app.pages.widgets.cart_item import CartItemWidget
from shopee_app.pages.widgets.cart_select_item import CartSelectItemWidget
from shopee_app.pages.widgets.product_card import ProductCard
from shopee_app.ui_gen.layout_user import Ui_Form_user as Ui_UserLayout
from shopee_app.pages.widgets.profile_dialog import ProfileDialog
from shopee_app.services.speech_to_text_worker import SpeechToTextWorker

class UserWindow(QWidget):

    closed = pyqtSignal()

    def __init__(
        self,
        *,
        user_info: dict[str, Any] | None = None,
        service_client: MainServiceClient | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.ui = Ui_UserLayout()
        self.ui.setupUi(self)

        self.setWindowTitle("Shopee GUI - User")
        self.products_container = getattr(self.ui, "grid_products", None)
        self.product_grid = getattr(self.ui, "gridLayout_2", None)
        self.products: list[ProductData] = []
        self.product_index: dict[int, ProductData] = {}
        self.default_empty_products_message = '표시할 상품이 없습니다.'
        self.empty_products_message = self.default_empty_products_message

        self.cart_items: dict[int, CartItemData] = {}
        self.cart_widgets: dict[int, CartItemWidget] = {}
        self.cart_items_layout = getattr(self.ui, "cart_items_layout", None)
        self.cart_spacer = None
        if self.cart_items_layout is not None:
            self.cart_spacer = QSpacerItem(
                0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
            )
            self.cart_items_layout.addItem(self.cart_spacer)
        self.ui.chk_select_all.stateChanged.connect(self.on_select_all_changed)
        if hasattr(self.ui, "btn_selected_delete"):
            self.ui.btn_selected_delete.clicked.connect(self.on_delete_selected_clicked)

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
        self.selection_buttons: list[QPushButton] = []
        self.setup_cart_section()
        self.setup_navigation()
        self.ui.btn_to_login_page.clicked.connect(self.close)
        self.ui.btn_pay.clicked.connect(self.on_pay_clicked)
        # 검색 입력 위젯을 저장하지 않으면 사용자가 입력한 검색어를 가져올 방법이 없다.
        self.search_input = getattr(self.ui, "edit_search", None)
        if self.search_input is None:
            self.search_input = getattr(self.ui, "lineEdit", None)
        # 위젯 존재 여부를 검증하지 않으면 None 객체에 연결을 시도해 런타임 오류가 발생한다.
        if self.search_input is not None:
            # 엔터 키 입력 시 검색을 자동으로 수행하지 않으면 사용자의 검색 흐름이 끊어진다.
            self.search_input.returnPressed.connect(self.on_search_submitted)
        self.search_button = getattr(self.ui, 'btn_search', None)
        if self.search_button is not None:
            # 버튼을 눌렀을 때 검색이 실행되지 않으면 사용자가 직관적으로 조작하기 어렵다.
            self.search_button.clicked.connect(self.on_search_button_clicked)
        self.mic_button = getattr(self.ui, 'btn_mic', None)
        if self.mic_button is not None:
            mic_icon_path = Path(__file__).resolve().parent / 'icons' / 'mic.svg'
            if mic_icon_path.exists():
                self.mic_button.setIcon(QIcon(str(mic_icon_path)))
                self.mic_button.setIconSize(QtCore.QSize(24, 24))
            self.mic_button.setText('')
            self.mic_button.setToolTip('음성으로 검색')
            self.mic_button.clicked.connect(self.on_microphone_clicked)
        self.mic_info_label = getattr(self.ui, 'label_mic_info', None)
        if self.mic_info_label is not None:
            self.mic_info_label.setText('')
            self.mic_info_label.setVisible(False)
        self._stt_feedback_timer = QtCore.QTimer(self)
        self._stt_feedback_timer.setInterval(1000)
        self._stt_feedback_timer.timeout.connect(self._on_stt_feedback_tick)
        self._stt_status_hide_timer = QtCore.QTimer(self)
        self._stt_status_hide_timer.setSingleShot(True)
        self._stt_status_hide_timer.timeout.connect(self._hide_mic_info)
        self._stt_countdown_seconds = 0
        self._stt_thread: QtCore.QThread | None = None
        self._stt_worker: SpeechToTextWorker | None = None
        self._stt_busy = False
        self._setup_speech_recognition()
        self.set_products(self.load_initial_products())
        self.update_cart_summary()

        self.user_info: dict[str, Any] = dict(user_info or {})
        self.service_client = (
            service_client if service_client is not None else MainServiceClient()
        )
        self.current_user_id = ''
        self._ensure_user_identity()
        self.current_order_id: int | None = None
        self.current_robot_id: int | None = None
        self.remote_selection_items: list[CartItemData] = []
        self.auto_selection_items: list[CartItemData] = []

        self.profile_dialog = ProfileDialog(self)
        self.profile_dialog.set_user_info(self.user_info)
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
        # 초기 화면에서 바로 서버 상품을 갱신하지 않으면 더미 데이터가 그대로 남는다.
        QtCore.QTimer.singleShot(0, self.request_total_products)

        self._update_user_header()
        self.notification_client: AppNotificationClient | None = None
        self._initialize_selection_grid()

    def _ensure_user_identity(self) -> str:
        # 로그인 여부와 관계없이 상품 검색을 테스트할 수 있도록 게스트 ID를 제공한다.
        user_id_value = ''
        if isinstance(self.user_info, dict):
            raw_id = self.user_info.get('user_id')
            if raw_id:
                user_id_value = str(raw_id).strip()
        if not user_id_value:
            fallback_user_id = os.getenv('SHOPEE_APP_GUEST_USER_ID', 'guest_user')
            user_id_value = fallback_user_id
            if isinstance(self.user_info, dict):
                self.user_info['user_id'] = user_id_value
        self.current_user_id = user_id_value
        return user_id_value

    def _setup_speech_recognition(self) -> None:
        if self._stt_worker is not None:
            return
        self._stt_thread = QtCore.QThread(self)
        stt_model_name = os.getenv('SHOPEE_STT_MODEL', 'base')
        self._stt_worker = SpeechToTextWorker(model_name=stt_model_name)
        self._stt_worker.moveToThread(self._stt_thread)
        self._stt_worker.started.connect(self.on_stt_started)
        self._stt_worker.result_ready.connect(self.on_stt_result_ready)
        self._stt_worker.error_occurred.connect(self.on_stt_error)
        self._stt_worker.finished.connect(self.on_stt_finished)
        self._stt_thread.start()

    def _shutdown_speech_recognition(self) -> None:
        if self._stt_thread is None:
            return
        self._stt_thread.quit()
        self._stt_thread.wait()
        self._stt_thread = None
        self._stt_worker = None
        self._stt_busy = False
        self._stt_feedback_timer.stop()
        self._stt_status_hide_timer.stop()
        if self.mic_button is not None:
            self.mic_button.setEnabled(True)
        self.unsetCursor()
        self._hide_mic_info()

    def closeEvent(self, event):
        if self.notification_client is not None:
            self.notification_client.stop()
        self._shutdown_speech_recognition()
        self.closed.emit()
        super().closeEvent(event)

    def on_pay_clicked(self):
        if self.request_create_order():
            self.set_mode("pick")

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

    def on_microphone_clicked(self) -> None:
        if self._stt_worker is None:
            QMessageBox.warning(
                self, '음성 인식 오류', '음성 인식 모듈이 초기화되지 않았습니다.'
            )
            return
        if self._stt_busy:
            QMessageBox.information(
                self, '음성 인식 진행 중', '이미 음성을 인식하고 있습니다.'
            )
            return
        self._start_mic_feedback()
        QtCore.QMetaObject.invokeMethod(
            self._stt_worker,
            'start_listening',
            QtCore.Qt.ConnectionType.QueuedConnection,
        )

    def on_stt_started(self) -> None:
        self._stt_busy = True
        if self.mic_button is not None:
            self.mic_button.setEnabled(False)
        self.setCursor(QtCore.Qt.CursorShape.WaitCursor)

    def on_stt_finished(self) -> None:
        self._stt_busy = False
        self._stt_feedback_timer.stop()
        if self.mic_button is not None:
            self.mic_button.setEnabled(True)
        self.unsetCursor()
        if not self._stt_status_hide_timer.isActive():
            self._hide_mic_info()

    def on_stt_result_ready(self, text: str) -> None:
        recognized = text.strip()
        if not recognized:
            self._show_mic_info('음성을 인식하지 못했습니다.')
            self._stt_status_hide_timer.start(2000)
            QMessageBox.information(self, '음성 인식', '음성을 인식하지 못했습니다.')
            return
        if self.search_input is not None:
            self.search_input.setText(recognized)
        self.request_product_search(recognized)
        self._show_mic_info(f'음성 인식 결과: {recognized}')
        self._stt_status_hide_timer.start(2500)
        QMessageBox.information(self, '음성 인식 결과', f'인식된 문장: {recognized}')

    def on_stt_error(self, message: str) -> None:
        QMessageBox.warning(self, '음성 인식 실패', message)
        self._show_mic_info('음성 인식 실패')
        self._stt_status_hide_timer.start(2500)

    def _start_mic_feedback(self) -> None:
        self._stt_status_hide_timer.stop()
        try:
            duration = float(os.getenv('SHOPEE_STT_FALLBACK_DURATION', '3.0'))
        except ValueError:
            duration = 3.0
        self._stt_countdown_seconds = max(1, math.ceil(duration))
        self._update_mic_info_label()
        self._stt_feedback_timer.start()

    def _on_stt_feedback_tick(self) -> None:
        if self._stt_countdown_seconds > 0:
            self._stt_countdown_seconds -= 1
        if self._stt_countdown_seconds > 0:
            self._update_mic_info_label()
            return
        self._stt_feedback_timer.stop()
        self._show_mic_info('음성 인식 중...')

    def _update_mic_info_label(self) -> None:
        if self.mic_info_label is None:
            return
        self.mic_info_label.setText(f'음성 인식 중... {self._stt_countdown_seconds}')
        self.mic_info_label.setVisible(True)

    def _show_mic_info(self, text: str) -> None:
        if self.mic_info_label is None:
            return
        self.mic_info_label.setText(text)
        self.mic_info_label.setVisible(True)

    def _hide_mic_info(self) -> None:
        if self.mic_info_label is None:
            return
        self.mic_info_label.setText('')
        self.mic_info_label.setVisible(False)

    def request_total_products(self) -> None:
        # 전체 목록을 가져오지 않으면 쇼핑 첫 화면에 표시할 데이터가 부족하다.
        if self.service_client is None:
            self.set_products(self.load_initial_products())
            self.refresh_product_grid()
            return
        user_id = self._ensure_user_identity()
        if not user_id:
            self.set_products(self.load_initial_products())
            self.refresh_product_grid()
            return
        try:
            response = self.service_client.fetch_total_products(user_id)
        except MainServiceClientError as exc:
            QMessageBox.warning(self, '상품 로드 실패', f'전체 상품을 불러오지 못했습니다.\n{exc}')
            self.set_products(self.load_initial_products())
            self.refresh_product_grid()
            return
        if not response:
            QMessageBox.warning(self, '상품 로드 실패', '서버에서 전체 상품 응답을 받지 못했습니다.')
            self.set_products(self.load_initial_products())
            self.refresh_product_grid()
            return
        if not response.get('result'):
            message = response.get('message') or '전체 상품을 가져오지 못했습니다.'
            QMessageBox.warning(self, '상품 로드 실패', message)
            self.set_products(self.load_initial_products())
            self.refresh_product_grid()
            return
        data = response.get('data') or {}
        entries = data.get('products') or []
        products = self._convert_total_products(entries)
        if not products:
            QMessageBox.information(self, '상품 로드 안내', '표시할 상품이 없어 기본 목록을 사용합니다.')
            self.set_products(self.load_initial_products())
            self.refresh_product_grid()
            return
        self.set_products(products)
        self.refresh_product_grid()

    def request_product_search(self, query: str) -> None:
        # 서비스 클라이언트가 없다면 네트워크 요청 자체가 불가능하므로 즉시 반환한다.
        if self.service_client is None:
            return
        user_id = self._ensure_user_identity()
        if not user_id:
            QMessageBox.warning(self, '검색 실패', '사용자 정보를 확인할 수 없습니다.')
            return
        # 사용자 정보를 기반으로 필터를 구성하지 않으면 개인화된 검색 조건이 적용되지 않는다.
        allergy_filter, vegan_flag = self._build_search_filter()
        # 검색 입력 위젯 참조를 보관하지 않으면 이후에 상태를 복원할 수 없다.
        search_widget = self.search_input
        # 위젯 존재 여부를 확인하지 않으면 None에 접근하면서 예외가 발생한다.
        if search_widget is not None:
            # 요청 동안 입력을 막지 않으면 사용자가 연타하여 중복 요청을 발생시킬 수 있다.
            search_widget.setEnabled(False)
        # 네트워크 예외를 처리하지 않으면 오류가 발생할 때 애플리케이션이 그대로 종료된다.
        try:
            # 명세에 맞춘 검색 요청을 호출하지 않으면 서버로부터 상품 목록을 받을 수 없다.
            response = self.service_client.search_products(
                user_id=user_id,
                query=query,
                allergy_filter=allergy_filter,
                is_vegan=vegan_flag,
            )
        except MainServiceClientError as exc:
            # 오류 알림을 하지 않으면 사용자가 검색 실패 원인을 알 수 없다.
            QMessageBox.warning(
                self, "검색 실패", f"상품 검색 중 오류가 발생했습니다.\n{exc}"
            )
            # 실패 시 기본 상품을 채워 넣지 않으면 화면이 비어 보이게 된다.
            self.set_products(self.load_initial_products())
            # 상품 목록을 다시 그리지 않으면 기존 화면이 갱신되지 않는다.
            self.refresh_product_grid()
            return
        finally:
            # 요청이 끝난 뒤 입력을 다시 활성화하지 않으면 사용자가 이후 검색을 할 수 없다.
            if search_widget is not None:
                search_widget.setEnabled(True)
        # 응답이 비어 있으면 이후 처리에서 KeyError가 발생할 수 있으므로 여기서 중단한다.
        if not response:
            QMessageBox.warning(self, "검색 실패", "서버에서 응답을 받지 못했습니다.")
            self.set_products(self.load_initial_products())
            self.refresh_product_grid()
            return
        # result 플래그를 확인하지 않으면 서버가 실패를 알린 경우에도 잘못된 데이터를 사용할 수 있다.
        if not response.get("result"):
            # 서버가 전달한 메시지를 표시하지 않으면 사용자가 실패 이유를 확인할 수 없다.
            message = response.get("message") or "상품을 불러오지 못했습니다."
            QMessageBox.warning(self, "검색 실패", message)
            self.set_products(self.load_initial_products())
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
            self.empty_products_message = '조건에 맞는 상품이 없습니다.'
            self.set_products([])
            self.refresh_product_grid()
            return
        # 변환된 상품을 상태에 반영하지 않으면 UI가 최신 정보를 표시하지 못한다.
        self.set_products(products)
        # 상품 목록을 다시 렌더링하지 않으면 화면에 여전히 이전 검색 결과가 남아 있다.
        self.refresh_product_grid()

    def _build_search_filter(self) -> tuple[dict[str, bool], bool | None]:
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

    def _convert_search_results(
        self, entries: list[dict[str, object]]
    ) -> list[ProductData]:
        # 결과를 누적할 리스트가 없으면 변환된 상품을 반환할 수 없다.
        products: list[ProductData] = []
        # 이미지 경로를 미리 정해두지 않으면 각 상품마다 반복 계산해야 한다.
        fallback_image = ProductCard.FALLBACK_IMAGE

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
                    image_path=fallback_image,
                )
            except TypeError:
                # 필수 필드가 누락된 경우 해당 상품만 건너뛰어 전체 처리를 계속한다.
                continue
            # 누락 없이 생성된 상품만 리스트에 추가한다.
            products.append(product)
        # 변환된 전체 목록을 반환하지 않으면 호출자가 결과를 사용할 수 없다.
        return products

    def _convert_total_products(self, entries: list[dict[str, object]]) -> list[ProductData]:
        # 전체 상품 응답이 비어 있으면 빈 리스트를 반환해야 이후 로직에서 목업 데이터를 사용할 수 있다.
        products: list[ProductData] = []
        fallback_image = ProductCard.FALLBACK_IMAGE
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                product_id = int(entry.get('product_id'))
            except (TypeError, ValueError):
                continue
            name = str(entry.get('name') or f'상품 {product_id}')
            category = str(entry.get('category') or '기타')
            price = int(entry.get('price') or 0)
            discount_rate = int(entry.get('discount_rate') or 0)
            is_vegan = bool(entry.get('is_vegan_friendly'))
            # total_product 응답에는 allergy_info_id, section_id 등이 없으므로 기본값을 채워 넣는다.
            product = ProductData(
                product_id=product_id,
                name=name,
                category=category,
                price=price,
                discount_rate=discount_rate,
                allergy_info_id=0,
                is_vegan_friendly=is_vegan,
                section_id=int(entry.get('section_id') or 0),
                warehouse_id=int(entry.get('warehouse_id') or 0),
                length=int(entry.get('length') or 0),
                width=int(entry.get('width') or 0),
                height=int(entry.get('height') or 0),
                weight=int(entry.get('weight') or 0),
                fragile=bool(entry.get('fragile')),
                image_path=fallback_image,
            )
            products.append(product)
        return products


    def setup_cart_section(self):
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

    def setup_navigation(self):
        self.main_stack = getattr(self.ui, "stacked_content", None)
        self.side_stack = getattr(self.ui, "stack_side_bar", None)
        self.page_user = getattr(self.ui, "page_content_user", None)
        self.page_pick = getattr(self.ui, "page_content_pick", None)
        self.side_shop_page = getattr(self.ui, "side_pick_page", None)
        self.side_pick_filter_page = getattr(self.ui, "side_allergy_filter_page", None)
        self.shopping_button = getattr(self.ui, "toolButton_3", None)
        self.store_button = getattr(self.ui, "toolButton_2", None)

        if self.shopping_button:
            self.shopping_button.setCheckable(True)
        if self.store_button:
            self.store_button.setCheckable(True)

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

    def on_cart_toggle_clicked(self):
        if self.cart_toggle_button is None:
            return

        self.cart_expanded = not self.cart_expanded
        self.apply_cart_state()

    def on_shopping_button_clicked(self):
        self.set_mode("shopping")

    def on_store_button_clicked(self):
        self.set_mode("pick")

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
            return self.handle_fake_order(selected_snapshot, selected_items)

        if not response:
            QMessageBox.warning(self, "주문 생성 실패", "서버 응답이 없습니다.")
            return self.handle_fake_order(selected_snapshot, selected_items)

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
        return self.handle_fake_order(selected_snapshot, selected_items)

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
        self._ensure_notification_listener()
        remote_items, auto_items = self.categorize_cart_items(ordered_items)
        self.remote_selection_items = remote_items
        self.auto_selection_items = auto_items

        remote_widget = getattr(self.ui, "widget_remote_select_list", None)
        auto_widget = getattr(self.ui, "widget_auto_select_list", None)
        self.populate_selection_list(remote_widget, remote_items, "원격 선택 대기")
        self.populate_selection_list(auto_widget, auto_items, "자동 선택 대기")

        self.current_order_id = order_data.get("order_id")
        self.current_robot_id = order_data.get("robot_id")

        status_label = getattr(self.ui, "label_12", None)
        if status_label is not None:
            if self.current_robot_id is not None:
                status_label.setText(f"로봇 {self.current_robot_id} 이동중")
            else:
                status_label.setText("로봇 이동중")

    def clear_ordered_cart_items(self, items: list[CartItemData]) -> None:
        for item in items:
            if item.product_id in self.cart_items:
                del self.cart_items[item.product_id]
            self.remove_cart_widget(item.product_id)
        self.update_cart_summary()
        self.sync_select_all_state()

    def handle_fake_order(
        self,
        ordered_snapshot: list[CartItemData],
        original_items: list[CartItemData],
    ) -> bool:
        QMessageBox.information(
            self,
            "임시 주문 진행",
            "Main Service 응답이 없어 임시 데이터로 화면을 전환합니다.",
        )
        self.handle_order_created(
            ordered_snapshot,
            {
                "order_id": -1,
                "robot_id": None,
            },
        )
        self.clear_ordered_cart_items(original_items)
        return True

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
        destination = data.get("destination")
        message_text = payload.get("message") or "로봇 이동중"
        status_label = getattr(self.ui, "label_12", None)
        footer_label = getattr(self.ui, "label_robot_notification", None)
        if destination:
            formatted = f"{message_text} : {destination}"
        else:
            formatted = message_text
        if status_label is not None:
            status_label.setText(formatted)
        if footer_label is not None:
            footer_label.setText(formatted)

        print(f"[알림] {formatted}")

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

        section_id = data.get("section_id")
        location_id = data.get("location_id")
        if isinstance(section_id, int) and section_id >= 0:
            location_text = f"SECTION_{section_id}"
        elif location_id is not None:
            location_text = f"LOCATION_{location_id}"
        else:
            location_text = None

        message_text = payload.get("message") or "로봇 도착"
        if location_text:
            formatted = f"{message_text} : {location_text}"
        else:
            formatted = message_text

        status_label = getattr(self.ui, "label_12", None)
        footer_label = getattr(self.ui, "label_robot_notification", None)
        if status_label is not None:
            status_label.setText(formatted)
        if footer_label is not None:
            footer_label.setText(formatted)
        print(f"[알림] {formatted}")

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
        footer_label = getattr(self.ui, "label_robot_notification", None)
        if status_label is not None:
            status_label.setText(message_text)
        if footer_label is not None:
            footer_label.setText(message_text)
        print(f"[알림] {message_text}")

    def handle_product_selection_start(self, payload: dict) -> None:
        """상품 선택 시작 알림을 처리한다."""
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

        products = data.get("products") or []
        product_names: list[str] = []
        for product in products:
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
        footer_label = getattr(self.ui, "label_robot_notification", None)
        if status_label is not None:
            status_label.setText(formatted)
        if footer_label is not None:
            footer_label.setText(formatted)
        self.populate_selection_buttons(products)
        print(f"[알림] {formatted}")

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

        if action == "add":
            default_message = "상품이 장바구니에 담겼습니다"
        elif action == "remove":
            default_message = "상품이 장바구니에서 제거되었습니다"
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
        footer_label = getattr(self.ui, "label_robot_notification", None)
        if status_label is not None:
            status_label.setText(formatted)
        if footer_label is not None:
            footer_label.setText(formatted)
        print(f"[알림] {formatted}")

    def _ensure_notification_listener(self) -> None:
        """주문 생성 시 알림 리스너를 시작한다."""
        if self.notification_client is not None:
            if not self.notification_client.isRunning():
                self.notification_client.start()
            return
        self.notification_client = AppNotificationClient(
            config=self.service_client.config
        )
        self.notification_client.notification_received.connect(
            self.on_notification_received
        )
        self.notification_client.connection_error.connect(self.on_notification_error)
        self.notification_client.start()

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
                widget.deleteLater()
        self.selection_buttons.clear()

        parent = (
            self.selection_container if self.selection_container is not None else self
        )
        for index, product in enumerate(products):
            button = QPushButton(parent)
            button.setFixedSize(123, 36)
            name = str(
                product.get("name")
                or product.get("product_id")
                or f"선택지 {index + 1}"
            )
            button.setText(name)
            button.setProperty("product_data", product)
            button.clicked.connect(
                lambda _, info=product: self.on_selection_button_clicked(info)
            )
            row, column = divmod(index, 5)
            self.selection_grid.addWidget(button, row, column)
            self.selection_buttons.append(button)
        for column in range(5):
            self.selection_grid.setColumnStretch(column, 1)

    def on_selection_button_clicked(self, product: dict[str, Any]) -> None:
        product_name = product.get("name") or product.get("product_id")
        QMessageBox.information(
            self, "상품 선택", f"{product_name} 선택지 버튼이 눌렸습니다."
        )

    def categorize_cart_items(
        self,
        items: list[CartItemData],
    ) -> tuple[list[CartItemData], list[CartItemData]]:
        # TODO: 서버에서 상품별 선택 방식 정보를 제공하면 해당 데이터를 사용한다.
        half = (len(items) + 1) // 2
        return items[:half], items[half:]

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
            widget.apply_item(
                index=index,
                name=item.name,
                quantity=item.quantity,
                status_text=status_text,
                image_path=item.image_path,
            )
            list_item = QListWidgetItem()
            list_item.setSizeHint(widget.sizeHint())
            list_widget.addItem(list_item)
            list_widget.setItemWidget(list_item, widget)

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
            self.show_main_page(self.page_user)
            self.show_side_page(self.side_pick_filter_page)
            return

        if mode == "pick":
            if self.store_button:
                self.store_button.setChecked(True)
            if self.shopping_button:
                self.shopping_button.setChecked(False)
            self.show_main_page(self.page_pick)
            self.show_side_page(self.side_shop_page)

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
                'empty_products_message',
                '표시할 상품이 없습니다.',
            )
            placeholder = QLabel(message)
            placeholder.setObjectName('product_grid_placeholder')
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
            card.apply_product(product)
            if hasattr(card.ui, "toolButton"):
                card.ui.toolButton.clicked.connect(
                    lambda _, p=product: self.on_add_to_cart(p)
                )
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
            label_amount.setText(f"{total_price:,}")

    def set_products(self, products: list[ProductData]) -> None:
        self.products = products
        self.product_index = {product.product_id: product for product in products}
        if products:
            self.empty_products_message = self.default_empty_products_message
        self.current_columns = -1

    def load_initial_products(self) -> list[ProductData]:
        image_root = Path(__file__).resolve().parent.parent / "image"
        return [
            ProductData(
                product_id=1,
                name="삼겹살",
                category="고기",
                price=15000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=True,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=2,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=3,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=4,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=2,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=2,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
            ProductData(
                product_id=2,
                name="서울우유",
                category="우유",
                price=1000,
                discount_rate=10,
                allergy_info_id=0,
                is_vegan_friendly=False,
                section_id=1,
                warehouse_id=1,
                length=20,
                width=15,
                height=5,
                weight=300,
                fragile=False,
                image_path=image_root / "product_no_image.png",
            ),
        ]
