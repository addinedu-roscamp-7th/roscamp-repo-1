import sys
from typing import Optional

from PyQt6.QtWidgets import QApplication

from shopee_app.pages.main_window import MainWindow
from shopee_app.ros_node import RosNodeThread


def create_application(argv: Optional[list[str]] = None) -> QApplication:
    '''PyQt 애플리케이션을 생성하고 기본 속성을 설정한다.'''
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName('Shopee App')
    return app


def main():
    '''메인 윈도우와 ROS 스레드를 초기화하고 이벤트 루프를 실행한다.'''
    app = create_application()
    ros_thread = RosNodeThread()
    window = MainWindow(ros_thread=ros_thread)
    ros_thread.node_error.connect(window.on_ros_error)
    ros_thread.node_ready.connect(window.on_ros_ready)
    app.aboutToQuit.connect(ros_thread.shutdown)
    window.show()
    ros_thread.start()
    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        ros_thread.shutdown()
        ros_thread.wait()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
