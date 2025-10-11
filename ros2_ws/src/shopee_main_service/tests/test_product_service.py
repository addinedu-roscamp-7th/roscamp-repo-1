"""
Unit tests for the ProductService.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from shopee_main_service.database_models import Product, Section, Shelf, Location
from shopee_main_service.product_service import ProductService

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_db_manager() -> MagicMock:
    """Fixture to create a mock DatabaseManager."""
    db_manager = MagicMock()
    db_manager.session_scope.return_value.__enter__.return_value = MagicMock()
    db_manager.session_scope.return_value.__exit__.return_value = None
    return db_manager


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Fixture to create a mock LLMClient."""
    return AsyncMock()


@pytest.fixture
def product_service(mock_db_manager: MagicMock, mock_llm_client: AsyncMock) -> ProductService:
    """Fixture to create a ProductService instance with mock dependencies."""
    return ProductService(db=mock_db_manager, llm_client=mock_llm_client)


class TestProductServiceSearch:
    """Test suite for the ProductService.search_products method."""

    def _create_mock_product(self, product_id, name):
        """Helper to create a mock product for testing."""
        mock_location = Location(location_id=1)
        mock_shelf = Shelf(shelf_id=1, location_id=1, location=mock_location)
        mock_section = Section(section_id=1, shelf_id=1, shelf=mock_shelf)
        return Product(
            product_id=product_id,
            name=name,
            barcode=f"barcode_{product_id}",
            quantity=10,
            price=1000,
            discount_rate=0,
            category="test",
            is_vegan_friendly=False,
            allergy_info_id=1,
            section_id=1,
            warehouse_id=1,
            section=mock_section
        )

    async def test_search_with_llm_success(self, product_service: ProductService, mock_db_manager: MagicMock, mock_llm_client: AsyncMock):
        """Test product search when the LLM successfully generates a query."""
        # Arrange
        query_text = "사과"
        llm_generated_clause = "name LIKE '%사과%'"
        mock_llm_client.generate_search_query.return_value = llm_generated_clause

        mock_product = self._create_mock_product(1, "청사과")
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.from_statement.return_value.all.return_value = [mock_product]

        # Act
        results = await product_service.search_products(query_text)

        # Assert
        mock_llm_client.generate_search_query.assert_awaited_once_with(query_text)
        mock_session.query.return_value.from_statement.assert_called_once()
        assert len(results) == 1
        assert results[0]["name"] == "청사과"

    async def test_search_with_llm_failure_fallback(self, product_service: ProductService, mock_db_manager: MagicMock, mock_llm_client: AsyncMock):
        """Test product search fallback mechanism when the LLM fails."""
        # Arrange
        query_text = "오렌지"
        mock_llm_client.generate_search_query.return_value = None  # Simulate LLM failure

        mock_product = self._create_mock_product(2, "오렌지")
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_product]

        # Act
        results = await product_service.search_products(query_text)

        # Assert
        mock_llm_client.generate_search_query.assert_awaited_once_with(query_text)
        mock_session.query.return_value.filter.assert_called_once()
        assert len(results) == 1
        assert results[0]["name"] == "오렌지"

    async def test_search_no_results(self, product_service: ProductService, mock_db_manager: MagicMock, mock_llm_client: AsyncMock):
        """Test product search that yields no results."""
        # Arrange
        query_text = "없는상품"
        mock_llm_client.generate_search_query.return_value = "name LIKE '%없는상품%'"
        
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.from_statement.return_value.all.return_value = []

        # Act
        results = await product_service.search_products(query_text)

        # Assert
        assert len(results) == 0
