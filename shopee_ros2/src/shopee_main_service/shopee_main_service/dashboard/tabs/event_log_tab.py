"""'이벤트 로그' 탭의 UI 로직"""
import csv
from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem, QFileDialog, QMessageBox, QHeaderView

from ..ui_gen.tab_event_log_ui import Ui_EventLogTab
from .base_tab import BaseTab


class EventLogTab(BaseTab, Ui_EventLogTab):
    """'이벤트 로그' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._all_events: List[Dict[str, Any]] = []
        self._filtered_events: List[Dict[str, Any]] = []
        
        # 테이블 컬럼 설정
        self._setup_table_columns()
        
        # 시그널 연결
        self.search_line_edit.textChanged.connect(self._apply_filters)
        self.level_filter_combo.currentTextChanged.connect(self._apply_filters)
        self.clear_button.clicked.connect(self._clear_logs)
        self.export_button.clicked.connect(self._export_csv)

    def _setup_table_columns(self):
        """테이블 컬럼 너비와 리사이즈 정책을 설정한다."""
        header = self.log_table.horizontalHeader()
        
        # 각 컬럼의 너비를 설정 (픽셀 단위)
        self.log_table.setColumnWidth(0, 80)   # Time
        self.log_table.setColumnWidth(1, 80)   # Level
        self.log_table.setColumnWidth(2, 150)  # Type
        
        # 마지막 컬럼(Message)은 남은 공간을 모두 차지하도록 설정
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        # 나머지 컬럼들은 고정 크기로 설정
        for i in range(3):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)

    def add_event(self, event_data: Dict[str, Any]):
        """새 이벤트를 로그에 추가한다."""
        timestamp = datetime.now()
        
        # 이벤트 레벨 결정
        event_type = event_data.get('type', 'unknown')
        level = 'INFO'
        if 'error' in event_type.lower() or 'fail' in event_type.lower():
            level = 'ERROR'
        elif 'warn' in event_type.lower() or 'timeout' in event_type.lower():
            level = 'WARN'
        
        event_entry = {
            'timestamp': timestamp,
            'level': level,
            'type': event_type,
            'message': event_data.get('message', ''),
            'metadata': event_data
        }
        
        self._all_events.insert(0, event_entry)
        
        # 최대 200개 유지
        if len(self._all_events) > 200:
            self._all_events = self._all_events[:200]
        
        # 필터 적용 및 테이블 업데이트
        self._apply_filters()

    def add_ros_topic_event(self, event_data: Dict[str, Any]):
        """ROS 토픽 이벤트를 로그에 추가한다."""
        timestamp = datetime.fromtimestamp(event_data.get('timestamp', datetime.now().timestamp()))
        topic_name = event_data.get('topic_name', 'unknown')
        msg_data = event_data.get('msg', {})
        
        # 토픽 이름에서 이벤트 타입 추출
        event_type = topic_name.replace('/', '').replace('_', ' ').title()
        
        # 메시지 내용을 간단히 요약
        message_parts = []
        if 'robot_id' in msg_data:
            message_parts.append(f"Robot {msg_data['robot_id']}")
        if 'order_id' in msg_data:
            message_parts.append(f"Order {msg_data['order_id']}")
        if 'location_id' in msg_data:
            message_parts.append(f"Location {msg_data['location_id']}")
        if 'success' in msg_data:
            success = msg_data['success'].lower() == 'true' if isinstance(msg_data['success'], str) else msg_data['success']
            message_parts.append('성공' if success else '실패')
        if 'message' in msg_data:
            message_parts.append(msg_data['message'])
        
        message = ' - '.join(message_parts) if message_parts else str(msg_data)
        
        # 이벤트 레벨 결정
        level = 'INFO'
        if 'error' in topic_name.lower() or 'fail' in topic_name.lower():
            level = 'ERROR'
        elif 'success' in msg_data and msg_data['success'].lower() == 'false':
            level = 'ERROR'
        elif 'timeout' in topic_name.lower() or 'warn' in topic_name.lower():
            level = 'WARN'
        
        event_entry = {
            'timestamp': timestamp,
            'level': level,
            'type': event_type,
            'message': message,
            'metadata': event_data
        }
        
        self._all_events.insert(0, event_entry)
        
        # 최대 200개 유지
        if len(self._all_events) > 200:
            self._all_events = self._all_events[:200]
        
        # 필터 적용 및 테이블 업데이트
        self._apply_filters()

    def update_data(self, data):
        """이 탭은 스냅샷 데이터를 사용하지 않고 이벤트만 받습니다."""
        pass

    def _apply_filters(self):
        """검색어와 레벨 필터를 적용한다."""
        search_text = self.search_line_edit.text().lower()
        level_filter = self.level_filter_combo.currentText()
        
        self._filtered_events = []
        
        for event in self._all_events:
            # 레벨 필터 적용
            if level_filter != '전체' and event['level'] != level_filter:
                continue
            
            # 검색어 필터 적용
            if search_text:
                searchable_text = f"{event['type']} {event['message']}".lower()
                if search_text not in searchable_text:
                    continue
            
            self._filtered_events.append(event)
        
        self._update_table()

    def _update_table(self):
        """테이블을 업데이트한다."""
        self.log_table.setRowCount(len(self._filtered_events))
        
        for row, event in enumerate(self._filtered_events):
            # 시간
            time_text = event['timestamp'].strftime('%H:%M:%S')
            self.log_table.setItem(row, 0, QTableWidgetItem(time_text))
            
            # 레벨 (색상 적용)
            level = event['level']
            level_item = QTableWidgetItem(level)
            if level == 'ERROR':
                level_item.setForeground(Qt.GlobalColor.red)
            elif level == 'WARN':
                level_item.setForeground(Qt.GlobalColor.darkYellow)
            else:
                level_item.setForeground(Qt.GlobalColor.blue)
            self.log_table.setItem(row, 1, level_item)
            
            # 타입
            self.log_table.setItem(row, 2, QTableWidgetItem(event['type']))
            
            # 메시지
            message = event['message'][:100] + '...' if len(event['message']) > 100 else event['message']
            self.log_table.setItem(row, 3, QTableWidgetItem(message))
        
        # 상태 라벨 업데이트
        total_count = len(self._all_events)
        filtered_count = len(self._filtered_events)
        self.status_label.setText(f'총 {filtered_count}건 표시 중 (전체: {total_count}건)')

    def _clear_logs(self):
        """로그를 지운다."""
        reply = QMessageBox.question(
            self, 
            '로그 지우기', 
            '모든 로그를 지우시겠습니까?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._all_events.clear()
            self._filtered_events.clear()
            self._update_table()

    def _export_csv(self):
        """로그를 CSV 파일로 내보낸다."""
        if not self._filtered_events:
            QMessageBox.information(self, '내보내기', '내보낼 로그가 없습니다.')
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'CSV 파일로 저장',
            f'event_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            'CSV Files (*.csv)'
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Time', 'Level', 'Type', 'Message'])
                    
                    for event in self._filtered_events:
                        writer.writerow([
                            event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                            event['level'],
                            event['type'],
                            event['message']
                        ])
                
                QMessageBox.information(self, '내보내기 완료', f'로그가 {file_path}에 저장되었습니다.')
            except Exception as e:
                QMessageBox.critical(self, '내보내기 실패', f'파일 저장 중 오류가 발생했습니다:\n{str(e)}')