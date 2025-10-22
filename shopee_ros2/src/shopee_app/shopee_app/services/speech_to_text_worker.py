from __future__ import annotations

import os
from typing import Any

from PyQt6 import QtCore

import sounddevice as sd
import torch
import whisper

from shopee_app.pages.STT_commu import Mike_Util
from shopee_app.pages.STT_commu import STT_Util


class SpeechToTextWorker(QtCore.QObject):

    started = QtCore.pyqtSignal()
    result_ready = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        model_name: str | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self.model_name = model_name if model_name else os.getenv('SHOPEE_STT_MODEL', 'base')
        self._mike_util = Mike_Util()
        self._stt_util = STT_Util()
        self._model: Any | None = None
        self._device_index: int | None = None
        self._sample_rate: int | None = None
        self._fp16 = torch.cuda.is_available()
        self._mutex = QtCore.QMutex()
        self._running = False

    @QtCore.pyqtSlot()
    def start_listening(self) -> None:
        if not self._acquire():
            self.error_occurred.emit('다른 음성 인식 작업이 진행 중입니다.')
            self.finished.emit()
            return

        self.started.emit()
        try:
            text = self._execute_stt()
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        else:
            self.result_ready.emit(text)
        finally:
            self._release()
            self.finished.emit()

    def _acquire(self) -> bool:
        # 동시에 두 번 실행되지 않도록 뮤텍스로 상태를 보호한다.
        locker = QtCore.QMutexLocker(self._mutex)
        if self._running:
            return False
        self._running = True
        locker.unlock()
        return True

    def _release(self) -> None:
        locker = QtCore.QMutexLocker(self._mutex)
        self._running = False
        locker.unlock()

    def _ensure_resources(self) -> None:
        if (
            self._model is not None
            and self._device_index is not None
            and self._sample_rate is not None
        ):
            return

        # 마이크 목록을 확인하지 않으면 어떤 장치를 사용할지 결정할 수 없다.
        mike_list = self._mike_util.mike_list_return()
        if not mike_list:
            raise RuntimeError('사용 가능한 마이크 장치를 찾을 수 없습니다.')

        device_index = self._mike_util.pick_mike_index(mike_list)
        if device_index is None:
            raise RuntimeError('사용 가능한 마이크 장치를 선택할 수 없습니다.')

        try:
            device_info = sd.query_devices(device_index)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError('오디오 장치 정보를 가져올 수 없습니다.') from exc

        sample_rate = device_info.get('default_samplerate') or 16000
        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError) as exc:
            raise RuntimeError('오디오 장치 샘플레이트를 해석할 수 없습니다.') from exc

        # Whisper 모델을 로드하지 않으면 음성을 텍스트로 변환할 수 없다.
        try:
            self._model = whisper.load_model(
                self.model_name,
                device='cuda' if self._fp16 else 'cpu',
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError('Whisper 모델을 불러오지 못했습니다.') from exc

        self._device_index = device_index
        self._sample_rate = sample_rate

    def _execute_stt(self) -> str:
        self._ensure_resources()
        if self._device_index is None or self._sample_rate is None or self._model is None:
            raise RuntimeError('음성 인식 자원이 준비되지 않았습니다.')
        primary_text = self._stt_util.stt_response(
            self._device_index,
            self._model,
            self._sample_rate,
            self._fp16,
        )
        if primary_text:
            return primary_text
        return self._run_fallback_capture()

    def _run_fallback_capture(self) -> str:
        # 에너지 기반 감지가 실패했을 때 고정 길이 샘플을 녹음하지 않으면 음성을 확보할 수 없다.
        if self._device_index is None or self._sample_rate is None or self._model is None:
            raise RuntimeError('음성 인식 자원이 준비되지 않았습니다.')
        duration_sec = float(os.getenv('SHOPEE_STT_FALLBACK_DURATION', '3.0'))
        try:
            recording = sd.rec(
                int(duration_sec * self._sample_rate),
                samplerate=self._sample_rate,
                channels=1,
                dtype='float32',
                device=self._device_index,
            )
            sd.wait()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError('마이크 입력을 읽어오지 못했습니다.') from exc
        flattened = recording.reshape(-1)
        if flattened.size == 0:
            return ''
        # Whisper는 16kHz 오디오를 기대하므로 리샘플을 하지 않으면 결과 품질이 크게 떨어진다.
        audio_16k = self._stt_util.resample_linear(flattened, self._sample_rate, 16000)
        result = self._model.transcribe(
            audio_16k,
            language=self._stt_util.language,
            fp16=self._fp16,
        )
        return (result.get('text') or '').strip()
