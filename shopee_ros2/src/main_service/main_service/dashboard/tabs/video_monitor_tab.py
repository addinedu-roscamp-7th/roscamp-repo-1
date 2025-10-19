'''
영상 모니터링 탭

로봇별 실시간 영상 스트림을 표시합니다.
'''
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, Optional

from PyQt6.QtCore import QTimer, pyqtSignal, pyqtSlot, QObject, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QScrollArea,
    QSizePolicy,
)

from ..ui_gen.tab_video_monitor_ui import Ui_VideoMonitorTab
from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class VideoFrameAssembler:
    '''
    UDP 청크를 재조립하여 완전한 프레임을 생성합니다.
    '''

    def __init__(self):
        self.frame_buffer: Dict[int, Dict[int, bytes]] = {}  # frame_id -> {chunk_idx: data}
        self.buffer_max_size = 10

    def add_chunk(self, frame_id: int, chunk_idx: int, total_chunks: int, data: bytes) -> Optional[bytes]:
        '''
        청크를 추가하고, 프레임이 완성되면 전체 데이터를 반환합니다.

        Returns:
            완성된 프레임 데이터 (bytes) 또는 None
        '''
        # 버퍼 크기 제한
        if len(self.frame_buffer) > self.buffer_max_size:
            oldest_frame = min(self.frame_buffer.keys())
            del self.frame_buffer[oldest_frame]

        # 청크 저장
        if frame_id not in self.frame_buffer:
            self.frame_buffer[frame_id] = {}
        self.frame_buffer[frame_id][chunk_idx] = data

        # 프레임 완성 확인
        if len(self.frame_buffer[frame_id]) == total_chunks:
            # 모든 청크를 순서대로 조립
            complete_frame = b''.join(
                self.frame_buffer[frame_id][i] for i in range(total_chunks)
            )
            # 완성된 프레임은 버퍼에서 제거
            del self.frame_buffer[frame_id]
            return complete_frame

        return None


class VideoMonitorTab(BaseTab, Ui_VideoMonitorTab):
    '''영상 모니터링 탭'''

    # Qt 시그널 정의 (스레드 간 통신용)
    packet_received = pyqtSignal(bytes)

    def __init__(self, streaming_service=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)  # .ui 파일 로드
        
        self._streaming_service = streaming_service

        # robot_id -> VideoFrameAssembler
        self._assemblers: Dict[int, VideoFrameAssembler] = defaultdict(VideoFrameAssembler)

        # robot_id -> 최신 QPixmap (영상 캐시)
        self._robot_pixmaps: Dict[int, QPixmap] = {}

        # robot_id -> 상태 정보 (FPS, Frame ID)
        self._robot_stats: Dict[int, dict] = {}

        # 현재 선택된 로봇 ID
        self._current_robot_id: Optional[int] = None

        # robot_id -> 마지막 프레임 시간 (FPS 계산용)
        self._last_frame_times: Dict[int, float] = {}

        # 영상 레이블 크기 정책 설정 (동적 리사이즈)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.video_label.setScaledContents(True)

        # 시그널-슬롯 연결
        self.robot_selector.currentIndexChanged.connect(self._on_robot_selected)
        self.refresh_button.clicked.connect(self._refresh_video_displays)
        self.packet_received.connect(self._handle_video_packet_qt)

        # StreamingService에 콜백 등록
        if streaming_service:
            streaming_service.set_dashboard_callback(self._on_video_packet)

    def update_data(self, data):
        '''데이터 업데이트 (실시간 스트리밍이므로 사용하지 않음)'''
        pass

    def _on_robot_selected(self, index: int):
        '''로봇 선택 이벤트: 선택된 로봇의 영상을 표시'''
        if index < 0:
            return
        
        # 선택된 로봇 ID 추출
        robot_text = self.robot_selector.itemText(index)
        if not robot_text or robot_text == '로봇을 선택하세요':
            self._current_robot_id = None
            self.video_label.clear()
            self.video_label.setText('로봇을 선택하세요')
            self.video_group.setTitle('🤖 로봇 선택 대기 중...')
            return
        
        # "Robot 1" 형식에서 ID 추출
        robot_id = int(robot_text.split()[1])
        self._current_robot_id = robot_id
        
        # 해당 로봇의 최신 영상 표시
        if robot_id in self._robot_pixmaps:
            self.video_label.setPixmap(self._robot_pixmaps[robot_id])
            stats = self._robot_stats.get(robot_id, {})
            fps = stats.get('fps', 0)
            frame_id = stats.get('frame_id', 0)
            self.video_group.setTitle(f'🤖 Robot {robot_id}  │  {fps:.1f} FPS  │  Frame: {frame_id}')
        else:
            self.video_label.clear()
            self.video_label.setText(f'Robot {robot_id} 영상 대기 중...')
            self.video_group.setTitle(f'🤖 Robot {robot_id}  │  대기 중...')

    def _refresh_video_displays(self):
        '''영상 표시 새로고침'''
        logger.info('Refreshing video displays')

    def _on_video_packet(self, packet: bytes):
        '''
        StreamingService 콜백: UDP 패킷 수신

        Qt 메인 스레드에서 호출되지 않을 수 있으므로,
        시그널을 emit하여 메인 스레드로 전달
        '''
        self.packet_received.emit(packet)

    @pyqtSlot(bytes)
    def _handle_video_packet_qt(self, packet: bytes):
        '''Qt 메인 스레드에서 실행되는 패킷 처리 (슬롯)'''
        self.handle_video_packet(packet)

    def add_robot_display(self, robot_id: int):
        '''
        로봇을 선택 목록에 추가
        
        새로운 로봇이 감지되면 콤보박스에 선택 항목을 추가합니다.
        실제 영상은 콤보박스에서 선택 시 표시됩니다.
        
        Args:
            robot_id: 로봇 식별자
        '''
        # 중복 체크: 이미 목록에 있으면 무시
        for i in range(self.robot_selector.count()):
            if self.robot_selector.itemText(i) == f'Robot {robot_id}':
                return

        # 콤보박스에 로봇 추가
        self.robot_selector.addItem(f'Robot {robot_id}')
        
        # 첫 번째 로봇이면 자동 선택
        if self.robot_selector.count() == 1:
            self.robot_selector.setCurrentIndex(0)

        logger.info(f'Added robot {robot_id} to selector')

    def handle_video_packet(self, packet: bytes):
        '''
        UDP 영상 패킷 처리

        Args:
            packet: 200바이트 JSON 헤더 + 이미지 데이터
        '''
        try:
            # JSON 헤더 파싱
            header_bytes = packet[:200]
            header_str = header_bytes.decode('utf-8').rstrip('\x00').strip()
            header = json.loads(header_str)

            robot_id = header.get('robot_id')
            frame_id = header.get('frame_id')
            chunk_idx = header.get('chunk_idx')
            total_chunks = header.get('total_chunks')
            data_size = header.get('data_size')

            if robot_id is None:
                logger.warning('Received packet without robot_id')
                return

            # 로봇이 선택 목록에 없으면 추가
            if robot_id not in self._robot_pixmaps:
                self.add_robot_display(robot_id)

            # 이미지 데이터 추출
            image_data = packet[200:200 + data_size]

            # 프레임 재조립
            assembler = self._assemblers[robot_id]
            complete_frame = assembler.add_chunk(frame_id, chunk_idx, total_chunks, image_data)

            if complete_frame:
                # 완성된 프레임을 저장하고, 현재 선택된 로봇이면 표시
                self._display_frame(robot_id, frame_id, complete_frame)

        except Exception as e:
            logger.error(f'Failed to handle video packet: {e}')

    def _display_frame(self, robot_id: int, frame_id: int, frame_data: bytes):
        '''
        프레임을 캐시에 저장하고, 현재 선택된 로봇이면 화면에 표시
        
        Args:
            robot_id: 로봇 식별자
            frame_id: 프레임 번호
            frame_data: JPEG 이미지 데이터
        '''
        try:
            # JPEG 데이터를 QImage로 로드
            image = QImage.fromData(frame_data, 'JPEG')
            if image.isNull():
                logger.warning(f'Invalid JPEG data for robot {robot_id}, frame {frame_id}')
                return

            # QPixmap으로 변환
            pixmap = QPixmap.fromImage(image)
            
            # FPS 계산
            fps = self._calculate_fps(robot_id, frame_id)
            
            # 캐시에 저장
            self._robot_pixmaps[robot_id] = pixmap
            self._robot_stats[robot_id] = {
                'fps': fps,
                'frame_id': frame_id
            }

            # 현재 선택된 로봇이면 화면에 표시
            if self._current_robot_id == robot_id:
                self.video_label.setPixmap(pixmap)
                self.video_group.setTitle(
                    f'🤖 Robot {robot_id}  │  {fps:.1f} FPS  │  Frame: {frame_id}'
                )

        except Exception as e:
            logger.error(f'Failed to display frame: {e}')

    def _calculate_fps(self, robot_id: int, frame_id: int) -> float:
        '''FPS 계산'''
        import time
        current_time = time.time()

        if robot_id in self._last_frame_times:
            elapsed = current_time - self._last_frame_times[robot_id]
            if elapsed > 0:
                fps = 1.0 / elapsed
                self._last_frame_times[robot_id] = current_time
                return fps

        self._last_frame_times[robot_id] = current_time
        return 10.0

    def cleanup(self):
        '''리소스 정리'''
        # 캐시 정리
        self._robot_pixmaps.clear()
        self._robot_stats.clear()
        self._assemblers.clear()
