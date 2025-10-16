"""
대시보드 GUI 런처

별도 스레드에서 PyQt6 GUI를 실행한다.
"""

import logging
import os
import sys
import threading

from PyQt6.QtWidgets import QApplication

from .window import DashboardWindow

logger = logging.getLogger(__name__)


def start_dashboard_gui(controller):
    """
    별도 스레드에서 Qt GUI를 실행한다.

    Args:
        controller: 초기화된 DashboardController 인스턴스
    """

    def gui_thread_main():
        """GUI 스레드 메인 함수"""
        try:
            # DISPLAY 환경변수가 없으면 offscreen 플랫폼 사용
            if not os.environ.get('DISPLAY'):
                logger.warning('No DISPLAY environment variable found. Using offscreen platform.')
                os.environ['QT_QPA_PLATFORM'] = 'offscreen'

            # Qt Application 생성
            # sys.argv를 전달하지 않고 빈 리스트 사용 (ROS2와 충돌 방지)
            app = QApplication([])

            # 메인 윈도우 생성
            window = DashboardWindow(controller.bridge)
            window.show()

            logger.info('Dashboard GUI started (platform: %s)', os.environ.get('QT_QPA_PLATFORM', 'default'))

            # Qt 이벤트 루프 실행
            app.exec()

            logger.info('Dashboard GUI closed')

        except Exception as exc:
            logger.exception('Dashboard GUI thread failed: %s', exc)

    # 데몬 스레드로 실행 (메인 프로세스 종료 시 자동 종료)
    thread = threading.Thread(target=gui_thread_main, daemon=True, name='DashboardGUI')
    thread.start()

    logger.info('Dashboard GUI thread launched')
