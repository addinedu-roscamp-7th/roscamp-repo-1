"""
Unit tests for the UserService.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from shopee_main_service.database_models import Customer
from shopee_main_service.user_service import UserService, pwd_context

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_db_manager() -> MagicMock:
    """Fixture to create a mock DatabaseManager."""
    db_manager = MagicMock()
    # session_scope 컨텍스트 매니저를 모킹합니다.
    db_manager.session_scope.return_value.__aenter__.return_value = MagicMock()
    db_manager.session_scope.return_value.__aexit__.return_value = None
    return db_manager


@pytest.fixture
def user_service(mock_db_manager: MagicMock) -> UserService:
    """Fixture to create a UserService instance with a mock DB manager."""
    return UserService(db=mock_db_manager)


class TestUserServiceLogin:
    """Test suite for the UserService.login method."""

    CORRECT_PASSWORD = "password123"
    # A pre-computed bcrypt hash for "password123"
    PRECOMPUTED_HASH = "$2b$12$EixZaYVK134s3Hn8s4nS.e1d.aa03q24f6v.k3l6p9j8s7J5k4L3i"

    @patch('shopee_main_service.user_service.pwd_context')
    async def test_login_success(self, mock_pwd_context, user_service: UserService, mock_db_manager: MagicMock):
        """Test successful login with correct credentials."""
        # Arrange
        mock_pwd_context.verify.return_value = True
        mock_customer = Customer(id="testuser", password=self.PRECOMPUTED_HASH)
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value # Use __enter__ for sync context manager
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_customer

        # Act
        result = await user_service.login(user_id="testuser", password=self.CORRECT_PASSWORD)

        # Assert
        assert result is True
        mock_pwd_context.verify.assert_called_once_with(self.CORRECT_PASSWORD, self.PRECOMPUTED_HASH)

    @patch('shopee_main_service.user_service.pwd_context')
    async def test_login_failure_wrong_password(self, mock_pwd_context, user_service: UserService, mock_db_manager: MagicMock):
        """Test failed login with an incorrect password."""
        # Arrange
        mock_pwd_context.verify.return_value = False
        mock_customer = Customer(id="testuser", password=self.PRECOMPUTED_HASH)
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_customer

        # Act
        result = await user_service.login(user_id="testuser", password="wrongpassword")

        # Assert
        assert result is False
        mock_pwd_context.verify.assert_called_once_with("wrongpassword", self.PRECOMPUTED_HASH)

    async def test_login_failure_user_not_found(self, user_service: UserService, mock_db_manager: MagicMock):
        """Test failed login when the user does not exist."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Act
        result = await user_service.login(user_id="nonexistentuser", password="anypassword")

        # Assert
        assert result is False
