"""
RobotHistoryService 단위 테스트

로봇 작업 이력 조회 기능을 테스트합니다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from main_service.database_models import RobotHistory
from main_service.robot_history_service import RobotHistoryService

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_db_manager() -> MagicMock:
    """Mock DatabaseManager fixture"""
    db_manager = MagicMock()
    mock_session = MagicMock()
    db_manager.session_scope.return_value.__enter__.return_value = mock_session
    db_manager.session_scope.return_value.__exit__.return_value = None
    return db_manager


@pytest.fixture
def robot_history_service(mock_db_manager: MagicMock) -> RobotHistoryService:
    """RobotHistoryService 인스턴스 생성"""
    return RobotHistoryService(db=mock_db_manager)


class TestRobotHistoryServiceSearchHistories:
    """로봇 이력 검색 테스트"""

    async def test_search_all_histories(
        self, robot_history_service: RobotHistoryService, mock_db_manager: MagicMock
    ):
        """전체 이력 조회"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        now = datetime.now(timezone.utc)
        mock_histories = [
            RobotHistory(
                robot_history_id=1,
                robot_id=1,
                order_item_id=100,
                failure_reason=None,
                is_complete=True,
                active_duration=5,
                created_at=now
            ),
            RobotHistory(
                robot_history_id=2,
                robot_id=2,
                order_item_id=101,
                failure_reason=None,
                is_complete=True,
                active_duration=7,
                created_at=now
            ),
        ]
        mock_session.query.return_value.all.return_value = mock_histories

        # Act
        histories, total = await robot_history_service.search_histories({})

        # Assert
        assert total == 2
        assert len(histories) == 2
        assert histories[0]['robot_history_id'] == 1
        assert histories[0]['robot_id'] == 1
        assert histories[1]['robot_history_id'] == 2

    async def test_search_by_robot_id(
        self, robot_history_service: RobotHistoryService, mock_db_manager: MagicMock
    ):
        """로봇 ID로 검색"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        now = datetime.now(timezone.utc)
        mock_history = RobotHistory(
            robot_history_id=1,
            robot_id=5,
            order_item_id=200,
            failure_reason=None,
            is_complete=True,
            active_duration=10,
            created_at=now
        )
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = [mock_history]
        mock_session.query.return_value = query_mock

        # Act
        histories, total = await robot_history_service.search_histories({'robot_id': 5})

        # Assert
        assert total == 1
        assert histories[0]['robot_id'] == 5
        assert histories[0]['order_item_id'] == 200
        query_mock.filter.assert_called_once()

    async def test_search_by_order_item_id(
        self, robot_history_service: RobotHistoryService, mock_db_manager: MagicMock
    ):
        """주문 아이템 ID로 검색"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        now = datetime.now(timezone.utc)
        mock_history = RobotHistory(
            robot_history_id=1,
            robot_id=3,
            order_item_id=300,
            failure_reason=None,
            is_complete=True,
            active_duration=8,
            created_at=now
        )
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = [mock_history]
        mock_session.query.return_value = query_mock

        # Act
        histories, total = await robot_history_service.search_histories({'order_item_id': 300})

        # Assert
        assert total == 1
        assert histories[0]['order_item_id'] == 300

    async def test_search_by_is_complete(
        self, robot_history_service: RobotHistoryService, mock_db_manager: MagicMock
    ):
        """완료 여부로 검색"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        now = datetime.now(timezone.utc)
        failed_history = RobotHistory(
            robot_history_id=1,
            robot_id=1,
            order_item_id=100,
            failure_reason='Robot error',
            is_complete=False,
            active_duration=3,
            created_at=now
        )
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = [failed_history]
        mock_session.query.return_value = query_mock

        # Act
        histories, total = await robot_history_service.search_histories({'is_complete': False})

        # Assert
        assert total == 1
        assert histories[0]['is_complete'] is False
        assert histories[0]['failure_reason'] == 'Robot error'

    async def test_search_by_failure_reason(
        self, robot_history_service: RobotHistoryService, mock_db_manager: MagicMock
    ):
        """실패 사유로 검색"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        now = datetime.now(timezone.utc)
        failed_history = RobotHistory(
            robot_history_id=1,
            robot_id=2,
            order_item_id=150,
            failure_reason='timeout',
            is_complete=False,
            active_duration=2,
            created_at=now
        )
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = [failed_history]
        mock_session.query.return_value = query_mock

        # Act
        histories, total = await robot_history_service.search_histories({'failure_reason': 'timeout'})

        # Assert
        assert total == 1
        assert histories[0]['failure_reason'] == 'timeout'

    async def test_search_by_active_duration(
        self, robot_history_service: RobotHistoryService, mock_db_manager: MagicMock
    ):
        """작업 시간으로 검색"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        now = datetime.now(timezone.utc)
        history = RobotHistory(
            robot_history_id=1,
            robot_id=1,
            order_item_id=100,
            failure_reason=None,
            is_complete=True,
            active_duration=15,
            created_at=now
        )
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = [history]
        mock_session.query.return_value = query_mock

        # Act
        histories, total = await robot_history_service.search_histories({'active_duration': 15})

        # Assert
        assert total == 1
        assert histories[0]['active_duration'] == 15

    async def test_search_by_created_at(
        self, robot_history_service: RobotHistoryService, mock_db_manager: MagicMock
    ):
        """생성 시간으로 검색"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        now = datetime.now(timezone.utc)
        history = RobotHistory(
            robot_history_id=1,
            robot_id=1,
            order_item_id=100,
            failure_reason=None,
            is_complete=True,
            active_duration=5,
            created_at=now
        )
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = [history]
        mock_session.query.return_value = query_mock

        # Act
        histories, total = await robot_history_service.search_histories({
            'created_at': now.isoformat()
        })

        # Assert
        assert total == 1
        query_mock.filter.assert_called()

    async def test_search_no_results(
        self, robot_history_service: RobotHistoryService, mock_db_manager: MagicMock
    ):
        """검색 결과 없음"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = []
        mock_session.query.return_value = query_mock

        # Act
        histories, total = await robot_history_service.search_histories({'robot_id': 999})

        # Assert
        assert total == 0
        assert len(histories) == 0


class TestRobotHistoryServiceHelpers:
    """헬퍼 메서드 테스트"""

    def test_history_to_dict(
        self, robot_history_service: RobotHistoryService
    ):
        """이력을 딕셔너리로 변환"""
        # Arrange
        now = datetime.now(timezone.utc)
        history = RobotHistory(
            robot_history_id=100,
            robot_id=5,
            order_item_id=200,
            failure_reason='test failure',
            is_complete=False,
            active_duration=12,
            created_at=now
        )

        # Act
        result = robot_history_service._history_to_dict(history)

        # Assert
        assert result['robot_history_id'] == 100
        assert result['robot_id'] == 5
        assert result['order_item_id'] == 200
        assert result['failure_reason'] == 'test failure'
        assert result['is_complete'] is False
        assert result['active_duration'] == 12
        assert result['created_at'] == now.isoformat()

    def test_history_to_dict_null_created_at(
        self, robot_history_service: RobotHistoryService
    ):
        """created_at이 None인 경우"""
        # Arrange
        history = RobotHistory(
            robot_history_id=1,
            robot_id=1,
            order_item_id=1,
            failure_reason=None,
            is_complete=True,
            active_duration=1,
            created_at=None
        )

        # Act
        result = robot_history_service._history_to_dict(history)

        # Assert
        assert result['created_at'] is None

    def test_parse_datetime_valid_iso(
        self, robot_history_service: RobotHistoryService
    ):
        """ISO 형식 문자열 파싱"""
        # Arrange
        iso_str = '2025-10-16T12:30:00+00:00'

        # Act
        result = robot_history_service._parse_datetime(iso_str)

        # Assert
        assert result is not None
        assert isinstance(result, datetime)

    def test_parse_datetime_with_z(
        self, robot_history_service: RobotHistoryService
    ):
        """Z로 끝나는 ISO 문자열 파싱"""
        # Arrange
        iso_str = '2025-10-16T12:30:00Z'

        # Act
        result = robot_history_service._parse_datetime(iso_str)

        # Assert
        assert result is not None

    def test_parse_datetime_invalid_string(
        self, robot_history_service: RobotHistoryService
    ):
        """잘못된 문자열"""
        # Arrange
        invalid_str = 'not a datetime'

        # Act
        result = robot_history_service._parse_datetime(invalid_str)

        # Assert
        assert result is None

    def test_parse_datetime_already_datetime(
        self, robot_history_service: RobotHistoryService
    ):
        """이미 datetime 객체인 경우"""
        # Arrange
        now = datetime.now(timezone.utc)

        # Act
        result = robot_history_service._parse_datetime(now)

        # Assert
        assert result == now

    def test_parse_datetime_none(
        self, robot_history_service: RobotHistoryService
    ):
        """None 값"""
        # Act
        result = robot_history_service._parse_datetime(None)

        # Assert
        assert result is None

    def test_to_bool_true_values(
        self, robot_history_service: RobotHistoryService
    ):
        """True로 변환되는 값들"""
        assert robot_history_service._to_bool(True) is True
        assert robot_history_service._to_bool('true') is True
        assert robot_history_service._to_bool('True') is True
        assert robot_history_service._to_bool('TRUE') is True
        assert robot_history_service._to_bool('1') is True
        assert robot_history_service._to_bool('yes') is True
        assert robot_history_service._to_bool('y') is True
        assert robot_history_service._to_bool(1) is True

    def test_to_bool_false_values(
        self, robot_history_service: RobotHistoryService
    ):
        """False로 변환되는 값들"""
        assert robot_history_service._to_bool(False) is False
        assert robot_history_service._to_bool('false') is False
        assert robot_history_service._to_bool('False') is False
        assert robot_history_service._to_bool('0') is False
        assert robot_history_service._to_bool('no') is False
        assert robot_history_service._to_bool(0) is False
        assert robot_history_service._to_bool('') is False
