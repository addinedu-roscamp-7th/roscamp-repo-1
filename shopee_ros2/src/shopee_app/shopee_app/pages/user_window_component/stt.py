from PyQt6 import QtCore
import sys
from typing import Callable
from typing import Any

from STT_module import STT_Module


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
