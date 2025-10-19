"""'DB 관리' 탭의 UI 로직"""
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog, QApplication,
    QMenu, QInputDialog, QAbstractItemView
)
from sqlalchemy import text

from ..ui_gen.tab_db_admin_ui import Ui_DBAdminTab
from .base_tab import BaseTab


class QueryWorker(QThread):
    """백그라운드에서 DB 쿼리를 실행하는 워커 스레드"""
    
    query_finished = pyqtSignal(bool, object)  # success, result_or_error
    
    def __init__(self, db_manager, query: str):
        super().__init__()
        self.db_manager = db_manager
        self.query = query.strip()
    
    def run(self):
        """쿼리를 실행하고 결과를 시그널로 전달한다."""
        try:
            with self.db_manager.session_scope() as session:
                # SELECT 또는 SHOW 쿼리인지 확인 (결과를 반환하는 쿼리)
                query_upper = self.query.upper().strip()
                is_select = query_upper.startswith('SELECT') or query_upper.startswith('SHOW') or query_upper.startswith('DESCRIBE') or query_upper.startswith('EXPLAIN')
                
                if is_select:
                    # SELECT 쿼리: 결과를 반환
                    result = session.execute(text(self.query))
                    rows = result.fetchall()
                    columns = list(result.keys()) if rows else []
                    self.query_finished.emit(True, {'rows': rows, 'columns': columns, 'type': 'select'})
                else:
                    # INSERT/UPDATE/DELETE 쿼리: 영향받은 행 수 반환
                    result = session.execute(text(self.query))
                    rowcount = result.rowcount
                    session.commit()
                    self.query_finished.emit(True, {'rowcount': rowcount, 'type': 'modify'})
                    
        except Exception as e:
            self.query_finished.emit(False, str(e))


class DBAdminTab(BaseTab, Ui_DBAdminTab):
    """'DB 관리' 탭의 UI 및 로직"""

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.db_manager = db_manager
        self.current_data: List[List[Any]] = []
        self.current_columns: List[str] = []
        self.current_table: Optional[str] = None  # 현재 조회 중인 테이블
        self.primary_key_column: Optional[str] = None  # 기본 키 컬럼
        self.query_worker: Optional[QueryWorker] = None
        
        # 테이블 컬럼 설정
        self._setup_table_columns()
        
        # 시그널 연결
        self.execute_button.clicked.connect(self._execute_query)
        self.clear_button.clicked.connect(self._clear_query)
        self.table_combo.currentTextChanged.connect(self._on_table_selected)
        self.show_tables_button.clicked.connect(self._show_tables)
        self.export_button.clicked.connect(self._export_csv)
        
        # 테이블 우클릭 메뉴 연결
        self.result_table.customContextMenuRequested.connect(self._show_context_menu)
        self.result_table.itemChanged.connect(self._on_item_changed)
        
        # 초기 상태 설정
        self.export_button.setEnabled(False)

        # 동적으로 테이블 목록 채우기
        self._populate_tables_combo()

    def _populate_tables_combo(self):
        """데이터베이스에서 테이블 목록을 가져와 콤보박스를 채운다."""
        try:
            with self.db_manager.session_scope() as session:
                result = session.execute(text("SHOW TABLES;"))
                tables = sorted([row[0] for row in result])
                
                # 기존 UI 파일에 하드코딩된 목록을 지우고 새로 채운다.
                self.table_combo.clear()
                self.table_combo.addItem("테이블 선택...")
                self.table_combo.addItems(tables)
        except Exception as e:
            # 실패 시 에러 메시지 표시
            self.table_combo.clear()
            self.table_combo.addItem("목록 로드 실패")
            QMessageBox.warning(self, '테이블 로드 오류', f'테이블 목록을 가져오는 데 실패했습니다:\n{e}')

    def _setup_table_columns(self):
        """결과 테이블의 기본 설정을 한다."""
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)

    def _execute_query(self):
        """SQL 쿼리를 실행한다."""
        query = self.query_text_edit.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, '경고', '쿼리를 입력해주세요.')
            return
        
        # 위험한 쿼리 확인
        dangerous_keywords = ['DROP', 'TRUNCATE', 'ALTER']
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                reply = QMessageBox.question(
                    self, 
                    '위험한 쿼리',
                    f'"__{keyword}__" 명령이 포함된 쿼리입니다.\n정말 실행하시겠습니까?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                break
        
        # UI 상태 변경
        self.execute_button.setEnabled(False)
        self.status_label.setText('쿼리 실행 중...')
        QApplication.processEvents()
        
        # 백그라운드에서 쿼리 실행
        self.query_worker = QueryWorker(self.db_manager, query)
        self.query_worker.query_finished.connect(self._on_query_finished)
        self.query_worker.finished.connect(self._on_worker_finished)
        self.query_worker.start()

    def _on_query_finished(self, success: bool, result: Any):
        """쿼리 실행 완료 시 호출되는 슬롯"""
        self.execute_button.setEnabled(True)
        
        if success:
            if result['type'] == 'select':
                # SELECT 쿼리 결과 표시
                rows = result['rows']
                columns = result['columns']
                self._display_query_result(rows, columns)
                self.status_label.setText(f'조회 완료: {len(rows)}행')
                self.export_button.setEnabled(len(rows) > 0)
            else:
                # INSERT/UPDATE/DELETE 결과 표시
                rowcount = result['rowcount']
                self._clear_result_table()
                self.status_label.setText(f'실행 완료: {rowcount}행 영향받음')
                self.export_button.setEnabled(False)
        else:
            # 오류 표시
            error_msg = str(result)
            QMessageBox.critical(self, '쿼리 오류', f'쿼리 실행 중 오류가 발생했습니다:\n\n{error_msg}')
            self.status_label.setText('오류 발생')
            self.export_button.setEnabled(False)
        
        # 워커 정리는 _on_worker_finished에서 처리

    def _on_worker_finished(self):
        """워커 스레드 완료 시 호출되는 슬롯"""
        if self.query_worker:
            self.query_worker.deleteLater()
            self.query_worker = None

    def _display_query_result(self, rows: List, columns: List[str]):
        """쿼리 결과를 테이블에 표시한다."""
        self.current_data = [list(row) for row in rows]
        self.current_columns = columns
        
        # 현재 테이블과 기본 키 추출
        self._detect_current_table_and_pk()
        
        # 테이블 설정
        self.result_table.setRowCount(len(rows))
        self.result_table.setColumnCount(len(columns))
        self.result_table.setHorizontalHeaderLabels(columns)
        
        # 데이터 채우기
        for row_idx, row_data in enumerate(rows):
            for col_idx, cell_data in enumerate(row_data):
                # None 값 처리
                display_text = str(cell_data) if cell_data is not None else 'NULL'
                item = QTableWidgetItem(display_text)
                
                # NULL 값은 회색으로 표시
                if cell_data is None:
                    item.setForeground(Qt.GlobalColor.gray)
                
                # 기본 키 컬럼은 편집 불가로 설정
                if self.primary_key_column and columns[col_idx] == self.primary_key_column:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setBackground(Qt.GlobalColor.lightGray)
                
                self.result_table.setItem(row_idx, col_idx, item)
        
        # 컬럼 너비 자동 조정
        self.result_table.resizeColumnsToContents()

    def _clear_result_table(self):
        """결과 테이블을 지운다."""
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)
        self.current_data.clear()
        self.current_columns.clear()

    def _clear_query(self):
        """쿼리 입력창을 지운다."""
        self.query_text_edit.clear()

    def _on_table_selected(self, table_name: str):
        """테이블 콤보박스에서 선택 시 호출된다."""
        if table_name and table_name != '테이블 선택...' and table_name != "목록 로드 실패":
            # MySQL 예약어 처리를 위해 백틱으로 감싸기
            query = f'SELECT * FROM `{table_name}` LIMIT 100;'
            self.query_text_edit.setPlainText(query)

    def _show_tables(self):
        """모든 테이블 목록을 조회한다."""
        query = 'SHOW TABLES;'
        self.query_text_edit.setPlainText(query)
        self._execute_query()

    def _export_csv(self):
        """결과를 CSV 파일로 내보낸다."""
        if not self.current_data:
            QMessageBox.information(self, '내보내기', '내보낼 데이터가 없습니다.')
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'CSV 파일로 저장',
            f'query_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            'CSV Files (*.csv)'
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # 헤더 작성
                    writer.writerow(self.current_columns)
                    
                    # 데이터 작성
                    for row in self.current_data:
                        # None 값을 빈 문자열로 변환
                        csv_row = ['' if cell is None else str(cell) for cell in row]
                        writer.writerow(csv_row)
                
                QMessageBox.information(self, '내보내기 완료', f'데이터가 {file_path}에 저장되었습니다.')
            except Exception as e:
                QMessageBox.critical(self, '내보내기 실패', f'파일 저장 중 오류가 발생했습니다:\n{str(e)}')

    def _detect_current_table_and_pk(self):
        """현재 쿼리에서 테이블명과 기본 키를 추출한다."""
        query = self.query_text_edit.toPlainText().strip().upper()
        
        # 단순한 SELECT FROM 패턴 매칭
        if 'FROM' in query:
            parts = query.split('FROM')[1].strip().split()
            if parts:
                table_name = parts[0].strip('`').lower()
                self.current_table = table_name
                
                # 테이블별 기본 키 매핑
                pk_mapping = {
                    'product': 'product_id',
                    'order': 'order_id',
                    'customer': 'customer_id',
                    'order_item': 'order_item_id',
                    'robot': 'robot_id',
                    'robot_history': 'robot_history_id',
                    'admin': 'admin_id',
                    'allergy_info': 'allergy_info_id',
                    'location': 'location_id',
                    'section': 'section_id',
                    'shelf': 'shelf_id',
                    'warehouse': 'warehouse_id',
                    'box': 'box_id'
                }
                
                self.primary_key_column = pk_mapping.get(table_name)
        else:
            self.current_table = None
            self.primary_key_column = None

    def _show_context_menu(self, position):
        """테이블에서 우클릭 시 컨텍스트 메뉴를 표시한다."""
        if not self.current_table or not self.primary_key_column:
            return
            
        item = self.result_table.itemAt(position)
        if not item:
            return
            
        menu = QMenu(self)
        
        # 행 삭제 액션
        delete_action = menu.addAction("이 행 삭제")
        delete_action.triggered.connect(lambda: self._delete_row(item.row()))
        
        # 새 행 추가 액션
        add_action = menu.addAction("새 행 추가")
        add_action.triggered.connect(self._add_new_row)
        
        menu.addSeparator()
        
        # 변경사항 저장 액션
        save_action = menu.addAction("변경사항 저장")
        save_action.triggered.connect(self._save_changes)
        
        # 변경사항 취소 액션
        refresh_action = menu.addAction("새로고침")
        refresh_action.triggered.connect(self._refresh_table)
        
        menu.exec(self.result_table.mapToGlobal(position))

    def _delete_row(self, row_index: int):
        """선택된 행을 삭제한다. 'order' 테이블의 경우, 관련된 'order_item'도 함께 삭제한다."""
        if not self.current_table or not self.primary_key_column:
            QMessageBox.warning(self, '경고', '삭제할 수 없습니다. 테이블 정보가 없습니다.')
            return

        pk_col_index = self.current_columns.index(self.primary_key_column)
        pk_value = self.current_data[row_index][pk_col_index]

        reply = QMessageBox.question(
            self,
            '행 삭제 확인',
            f'정말로 이 행을 삭제하시겠습니까?\n{self.primary_key_column}: {pk_value}',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                with self.db_manager.session_scope() as session:
                    # 'order' 테이블을 삭제하는 경우, 관련된 'order_item'을 먼저 삭제
                    if self.current_table == 'order':
                        # 1. order_item 삭제
                        order_item_query = text("DELETE FROM `order_item` WHERE `order_id` = :pk_value")
                        session.execute(order_item_query, {"pk_value": pk_value})

                    # 2. 원본 행 삭제
                    query = text(f"DELETE FROM `{self.current_table}` WHERE `{self.primary_key_column}` = :pk_value")
                    result = session.execute(query, {"pk_value": pk_value})
                    
                    session.commit()
                    
                    self.status_label.setText(f"행이 삭제되었습니다. (영향받은 행: {result.rowcount})")
                    self._refresh_table() # 테이블 새로고침

            except Exception as e:
                QMessageBox.critical(self, '쿼리 오류', f'쿼리 실행 중 오류가 발생했습니다:\n{str(e)}')
                self.status_label.setText('오류 발생')

    def _add_new_row(self):
        """새 행을 추가한다."""
        if not self.current_table:
            QMessageBox.warning(self, '경고', '새 행을 추가할 수 없습니다. 테이블 정보가 없습니다.')
            return
            
        row_count = self.result_table.rowCount()
        self.result_table.insertRow(row_count)
        
        for col_idx in range(len(self.current_columns)):
            item = QTableWidgetItem("")
            if self.primary_key_column and self.current_columns[col_idx] == self.primary_key_column:
                item.setText("AUTO")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setBackground(Qt.GlobalColor.lightGray)
            self.result_table.setItem(row_count, col_idx, item)
        
        new_row = [None] * len(self.current_columns)
        self.current_data.append(new_row)

    def _on_item_changed(self, item):
        """테이블 아이템이 변경되었을 때 호출된다."""
        row = item.row()
        col = item.column()
        new_value_str = item.text()
        
        if row < len(self.current_data) and col < len(self.current_data[row]):
            final_value: Any
            if new_value_str.upper() == 'NULL' or new_value_str == '':
                final_value = None
                item.setForeground(Qt.GlobalColor.gray)
            else:
                # Try to convert to a number, otherwise keep as string
                try:
                    final_value = int(new_value_str)
                except ValueError:
                    try:
                        final_value = float(new_value_str)
                    except ValueError:
                        final_value = new_value_str  # It's a string
                
                item.setForeground(Qt.GlobalColor.black)

            self.current_data[row][col] = final_value

    def _save_changes(self):
        """변경사항을 데이터베이스에 저장한다."""
        if not self.current_table or not self.primary_key_column:
            QMessageBox.warning(self, '경고', '저장할 수 없습니다. 테이블 정보가 없습니다.')
            return
            
        try:
            pk_col_index = self.current_columns.index(self.primary_key_column)
            
            for row_idx, row_data in enumerate(self.current_data):
                pk_value = row_data[pk_col_index]
                
                if pk_value is None or str(pk_value).upper() == 'AUTO':
                    self._insert_new_row(row_data, row_idx)
                else:
                    self._update_existing_row(row_data, pk_value)
                    
            QMessageBox.information(self, '저장 완료', '모든 변경사항이 저장되었습니다.')
            self._refresh_table()
            
        except Exception as e:
            QMessageBox.critical(self, '저장 오류', f'저장 중 오류가 발생했습니다:\n{str(e)}')

    def _insert_new_row(self, row_data: List, row_idx: int):
        """새 행을 데이터베이스에 삽입한다."""
        pk_col_index = self.current_columns.index(self.primary_key_column)
        
        columns_to_insert = []
        params = {}
        
        for col_idx, (col_name, value) in enumerate(zip(self.current_columns, row_data)):
            if col_idx == pk_col_index:
                continue
            columns_to_insert.append(col_name)
            params[col_name] = value
        
        column_names = ", ".join([f"`{col}`" for col in columns_to_insert])
        param_names = ", ".join([f":{col}" for col in columns_to_insert])
        
        query = text(f"INSERT INTO `{self.current_table}` ({column_names}) VALUES ({param_names})")
        self._execute_update_query(query, params, "새 행이 추가되었습니다.")

    def _update_existing_row(self, row_data: List, pk_value):
        """기존 행을 업데이트한다."""
        pk_col_index = self.current_columns.index(self.primary_key_column)
        
        set_clauses = []
        params = {}
        for col_idx, (col_name, value) in enumerate(zip(self.current_columns, row_data)):
            if col_idx == pk_col_index:
                continue
            set_clauses.append(f"`{col_name}` = :{col_name}")
            params[col_name] = value
        
        params[self.primary_key_column] = pk_value
        
        query = text(f"UPDATE `{self.current_table}` SET { ', '.join(set_clauses)} WHERE `{self.primary_key_column}` = :{self.primary_key_column}")
        self._execute_update_query(query, params, "행이 업데이트되었습니다.")

    def _execute_update_query(self, query: text, params: Dict[str, Any], success_message: str):
        """업데이트 쿼리를 실행한다."""
        try:
            with self.db_manager.session_scope() as session:
                result = session.execute(query, params)
                session.commit()
                self.status_label.setText(f"{success_message} (영향받은 행: {result.rowcount})")
        except Exception as e:
            QMessageBox.critical(self, '쿼리 오류', f'쿼리 실행 중 오류가 발생했습니다:\n{str(e)}')

    def _refresh_table(self):
        """현재 테이블을 새로고침한다."""
        if self.current_table:
            query = f"SELECT * FROM `{self.current_table}`"
            self.query_text_edit.setPlainText(query)
            self._execute_query()

    def cleanup(self):
        """탭 종료 시 리소스 정리"""
        if self.query_worker and self.query_worker.isRunning():
            self.query_worker.terminate()
            self.query_worker.wait(3000)  # 3초 대기
            if self.query_worker.isRunning():
                self.query_worker.kill()
            self.query_worker = None

    def update_data(self, data):
        """이 탭은 스냅샷 데이터를 사용하지 않습니다."""
        pass
