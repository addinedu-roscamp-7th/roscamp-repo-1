from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

from PyQt6 import QtCore
from PyQt6.QtGui import QPainter
from PyQt6.QtGui import QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QMessageBox

from shopee_app.pages.admin_window import AdminWindow
from shopee_app.pages.user_window import UserWindow
from shopee_app.ui_gen.main_window import Ui_MainWindow
from shopee_app.services.main_service_client import MainServiceClient
from shopee_app.services.main_service_client import MainServiceClientError

if TYPE_CHECKING:
    from shopee_app.ros_node import RosNodeThread


class MainWindow(QMainWindow):
    def __init__(self, ros_thread: Optional["RosNodeThread"] = None):
        super().__init__()
        self._ros_thread = ros_thread
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Shopee GUI (PyQt6)")
        self.service_client = MainServiceClient()
        self._user_info: dict[str, Any] | None = None
        self._pixmap_helpers: list[_AspectRatioPixmapHelper] = []
        QtCore.QTimer.singleShot(0, self._apply_brand_images)
        self.setup_role_toggle()
        self.child_window = None
        self.ui.btn_login.clicked.connect(self.on_login_clicked)

    def setup_role_toggle(self):
        self.ui.btn_user.setCheckable(True)
        self.ui.btn_admin.setCheckable(True)
        self.role_group = QButtonGroup(self)
        self.role_group.setExclusive(True)
        self.role_group.addButton(self.ui.btn_user)
        self.role_group.addButton(self.ui.btn_admin)
        self.ui.btn_user.setChecked(True)

    def on_login_clicked(self):
        user_id = self.ui.et_id.text().strip()
        password = self.ui.et_password.text().strip()

        if not user_id or not password:
            QMessageBox.warning(self, '로그인 실패', '아이디와 비밀번호를 모두 입력해주세요.')
            return

        self.ui.btn_login.setEnabled(False)
        try:
            response = self.service_client.login(user_id, password)
        except MainServiceClientError as exc:
            QMessageBox.warning(self, '서버 연결 실패', f'{exc}\\n임시 계정으로 계속 진행합니다.')
            response = {
                'result': True,
                'data': {
                    'user_id': user_id,
                    'name': user_id or '임시 사용자',
                },
            }

        self.ui.btn_login.setEnabled(True)

        if not response or not response.get('result'):
            message = response.get('message') or '로그인에 실패했습니다.'
            QMessageBox.warning(self, '로그인 실패', message)
            return

        user_info = response.get('data') or {}
        user_info.setdefault('user_id', user_id)
        # 알림 전용 연결에서도 재인증할 수 있도록 비밀번호를 보관한다.
        user_info.setdefault('password', password)
        self._user_info = user_info
        self.ui.et_password.clear()

        if self.ui.btn_user.isChecked():
            self.launch_role_window(
                lambda: UserWindow(
                    user_info=user_info,
                    service_client=self.service_client,
                )
            )
            return

        if self.ui.btn_admin.isChecked():
            self.launch_role_window(lambda: AdminWindow())
            return

    def launch_role_window(self, factory):
        if self.child_window:
            self.child_window.close()
            self.child_window = None

        self.child_window = factory()
        self.child_window.show()
        self.hide()
        self.child_window.closed.connect(self.on_child_closed)

    def on_child_closed(self):
        self.child_window = None
        self.show()

    def on_ros_ready(self):
        # TODO: ROS2 데이터 연동 후 초기 화면 업데이트 로직을 추가한다.
        pass

    def on_ros_error(self, message: str):
        QMessageBox.critical(self, "ROS2 오류", message)
        self.close()

    def _apply_brand_images(self) -> None:
        """로그인 화면의 로고와 메인 이미지를 설정한다."""
        base_dir = Path(__file__).resolve().parent / "image"
        self._set_svg_to_label(
            getattr(self.ui, "label_logo", None),
            base_dir / "logo.svg",
            QtCore.QSize(180, 70),
        )
        self._set_png_to_label(
            getattr(self.ui, "label_main_image", None),
            base_dir / "main_image.png",
            QtCore.QSize(400, 300),
        )
        container = getattr(self.ui, "widget_main_image", None)
        if container is not None:
            container.setStyleSheet("background-color: #ff4649; border-radius: 30px;")

    def _set_svg_to_label(
        self,
        label,
        svg_path: Path,
        forced_size: QtCore.QSize | None,
    ) -> None:
        """SVG 파일을 QLabel에 렌더링한다."""
        if label is None or not svg_path.exists():
            return

        renderer = QSvgRenderer(str(svg_path))
        if forced_size is not None:
            target_size = forced_size
        else:
            target_size = label.size()
            if target_size.width() <= 0 or target_size.height() <= 0:
                hint = label.sizeHint()
                minimum = label.minimumSize()
                default = renderer.defaultSize()
                width = max(
                    target_size.width(),
                    minimum.width(),
                    hint.width(),
                    default.width() or 200,
                )
                height = max(
                    target_size.height(),
                    minimum.height(),
                    hint.height(),
                    default.height() or 200,
                )
                target_size = QtCore.QSize(width, height)

        pixmap = QPixmap(target_size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()

        self._register_pixmap_helper(label, pixmap, forced_size)

    def _set_png_to_label(
        self,
        label,
        png_path: Path,
        forced_size: QtCore.QSize | None,
    ) -> None:
        """PNG 이미지를 QLabel에 설정한다."""
        if label is None or not png_path.exists():
            return

        pixmap = QPixmap(str(png_path))
        if pixmap.isNull():
            return

        self._register_pixmap_helper(label, pixmap, forced_size)

    def _register_pixmap_helper(
        self,
        label,
        pixmap: QPixmap,
        forced_size: QtCore.QSize | None,
    ) -> None:
        """라벨에 픽스맵을 적용하고 비율 유지를 위한 헬퍼를 등록한다."""
        helper = _AspectRatioPixmapHelper(label, pixmap, forced_size)
        self._pixmap_helpers.append(helper)


class _AspectRatioPixmapHelper(QtCore.QObject):
    """QLabel에 설정된 픽스맵의 종횡비를 유지하며 리사이즈에 대응한다."""

    def __init__(
        self,
        label,
        pixmap: QPixmap,
        forced_size: QtCore.QSize | None = None,
    ) -> None:
        super().__init__(label)
        self._label = label
        self._pixmap = pixmap
        self._forced_size = forced_size

        if self._forced_size is not None:
            self._label.setMinimumSize(self._forced_size)
        self._label.setScaledContents(False)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.installEventFilter(self)
        self._update_pixmap()

    def eventFilter(self, obj, event):
        if obj is self._label and event.type() == QtCore.QEvent.Type.Resize:
            self._update_pixmap()
        return super().eventFilter(obj, event)

    def _update_pixmap(self) -> None:
        if self._pixmap.isNull():
            return

        label_size = self._label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            label_size = self._pixmap.size()

        if self._forced_size is not None:
            target_width = min(label_size.width(), self._forced_size.width())
            target_height = min(label_size.height(), self._forced_size.height())
            target_size = QtCore.QSize(target_width, target_height)
        else:
            target_size = label_size

        scaled = self._pixmap.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
