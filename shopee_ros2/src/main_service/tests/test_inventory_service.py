"""
InventoryService 단위 테스트

재고 관리 기능을 테스트합니다.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from main_service.database_models import Product, Section, Warehouse, Location, Shelf
from main_service.inventory_service import InventoryService

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
def inventory_service(mock_db_manager: MagicMock) -> InventoryService:
    """InventoryService 인스턴스 생성"""
    return InventoryService(db=mock_db_manager)


class TestInventoryServiceSearchProducts:
    """상품 검색 테스트"""

    async def test_search_all_products(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """전체 상품 조회"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_products = [
            Product(product_id=1, barcode='001', name='사과', quantity=10, price=1000,
                    section_id=1, category='fruit', allergy_info_id=1, is_vegan_friendly=True),
            Product(product_id=2, barcode='002', name='바나나', quantity=20, price=2000,
                    section_id=1, category='fruit', allergy_info_id=1, is_vegan_friendly=True),
        ]
        mock_session.query.return_value.all.return_value = mock_products

        # Act
        products, total = await inventory_service.search_products({})

        # Assert
        assert total == 2
        assert len(products) == 2
        assert products[0]['product_id'] == 1
        assert products[0]['name'] == '사과'
        assert products[1]['product_id'] == 2

    async def test_search_by_product_id(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """상품 ID로 검색"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_product = Product(
            product_id=42, barcode='042', name='테스트 상품', quantity=5, price=3000,
            section_id=2, category='test', allergy_info_id=2, is_vegan_friendly=False
        )
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = [mock_product]
        mock_session.query.return_value = query_mock

        # Act
        products, total = await inventory_service.search_products({'product_id': 42})

        # Assert
        assert total == 1
        assert products[0]['product_id'] == 42
        assert products[0]['name'] == '테스트 상품'

    async def test_search_by_name(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """상품명으로 검색 (부분 일치)"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_products = [
            Product(product_id=1, barcode='001', name='청사과', quantity=10, price=1000,
                    section_id=1, category='fruit', allergy_info_id=1, is_vegan_friendly=True),
            Product(product_id=2, barcode='002', name='홍사과', quantity=15, price=1200,
                    section_id=1, category='fruit', allergy_info_id=1, is_vegan_friendly=True),
        ]
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = mock_products
        mock_session.query.return_value = query_mock

        # Act
        products, total = await inventory_service.search_products({'name': '사과'})

        # Assert
        assert total == 2
        query_mock.filter.assert_called_once()

    async def test_search_by_quantity_range(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """재고 수량 범위로 검색"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_product = Product(
            product_id=1, barcode='001', name='사과', quantity=15, price=1000,
            section_id=1, category='fruit', allergy_info_id=1, is_vegan_friendly=True
        )
        query_mock = MagicMock()
        query_mock.filter.return_value.all.return_value = [mock_product]
        mock_session.query.return_value = query_mock

        # Act (quantity: 10 ~ 20)
        products, total = await inventory_service.search_products({'quantity': [10, 20]})

        # Assert
        assert total == 1
        assert products[0]['quantity'] == 15


class TestInventoryServiceCreateProduct:
    """상품 추가 테스트"""

    async def test_create_product_success(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """상품 추가 성공"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Mock section & warehouse resolution
        mock_section = Section(section_id=1, shelf_id=1)
        mock_shelf = Shelf(shelf_id=1, location_id=10)
        mock_section.shelf = mock_shelf
        mock_warehouse = Warehouse(warehouse_id=1, location_id=10)

        def query_side_effect(model):
            if model == Section:
                return MagicMock(filter_by=MagicMock(return_value=MagicMock(first=MagicMock(return_value=mock_section))))
            if model == Warehouse:
                return MagicMock(filter_by=MagicMock(return_value=MagicMock(first=MagicMock(return_value=mock_warehouse))))
            return MagicMock(filter_by=MagicMock(return_value=MagicMock(first=MagicMock(return_value=None))))

        mock_session.query.side_effect = query_side_effect

        payload = {
            'product_id': 100,
            'barcode': '1234567890',
            'name': '신상품',
            'quantity': 50,
            'price': 5000,
            'section_id': 1,
            'category': 'new',
            'allergy_info_id': 1,
            'is_vegan_friendly': True,
        }

        # Act
        await inventory_service.create_product(payload)

        # Assert
        mock_session.add.assert_called_once()
        added_product = mock_session.add.call_args[0][0]
        assert isinstance(added_product, Product)
        assert added_product.product_id == 100
        assert added_product.name == '신상품'

    async def test_create_product_duplicate(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """중복 상품 추가 시 에러"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        existing_product = Product(product_id=100, barcode='000', name='기존상품', quantity=1, price=1,
                                   section_id=1, category='a', allergy_info_id=1, is_vegan_friendly=True)
        mock_session.query.return_value.filter_by.return_value.first.return_value = existing_product

        payload = {
            'product_id': 100,
            'barcode': '1234567890',
            'name': '신상품',
            'quantity': 50,
            'price': 5000,
            'section_id': 1,
            'category': 'new',
            'allergy_info_id': 1,
            'is_vegan_friendly': True,
        }

        # Act & Assert
        with pytest.raises(ValueError, match='already exists'):
            await inventory_service.create_product(payload)

    async def test_create_product_missing_fields(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """필수 필드 누락 시 에러"""
        # Arrange
        payload = {
            'product_id': 100,
            'name': '불완전한 상품',
            # 나머지 필드 누락
        }

        # Act & Assert
        with pytest.raises(ValueError, match='Missing fields'):
            await inventory_service.create_product(payload)


class TestInventoryServiceUpdateProduct:
    """상품 수정 테스트"""

    async def test_update_product_success(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """상품 수정 성공"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        existing_product = Product(
            product_id=1, barcode='001', name='구상품', quantity=10, price=1000,
            section_id=1, category='old', allergy_info_id=1, is_vegan_friendly=False, warehouse_id=1
        )
        mock_session.query.return_value.filter_by.return_value.first.return_value = existing_product

        payload = {
            'product_id': 1,
            'name': '신상품',
            'price': 2000,
            'quantity': 20,
        }

        # Act
        result = await inventory_service.update_product(payload)

        # Assert
        assert result is True
        assert existing_product.name == '신상품'
        assert existing_product.price == 2000
        assert existing_product.quantity == 20

    async def test_update_product_not_found(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """존재하지 않는 상품 수정"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        payload = {'product_id': 999, 'name': '없는상품'}

        # Act
        result = await inventory_service.update_product(payload)

        # Assert
        assert result is False

    async def test_update_product_missing_id(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """product_id 없이 수정 시도"""
        # Arrange
        payload = {'name': '상품명만'}

        # Act & Assert
        with pytest.raises(ValueError, match='product_id is required'):
            await inventory_service.update_product(payload)


class TestInventoryServiceDeleteProduct:
    """상품 삭제 테스트"""

    async def test_delete_product_success(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """상품 삭제 성공"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        existing_product = Product(
            product_id=1, barcode='001', name='삭제할상품', quantity=10, price=1000,
            section_id=1, category='test', allergy_info_id=1, is_vegan_friendly=True
        )
        mock_session.query.return_value.filter_by.return_value.first.return_value = existing_product

        # Act
        result = await inventory_service.delete_product(1)

        # Assert
        assert result is True
        mock_session.delete.assert_called_once_with(existing_product)

    async def test_delete_product_not_found(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """존재하지 않는 상품 삭제"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Act
        result = await inventory_service.delete_product(999)

        # Assert
        assert result is False
        mock_session.delete.assert_not_called()


class TestInventoryServiceStockManagement:
    """재고 예약/해제 테스트"""

    async def test_check_and_reserve_stock_success(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """재고 예약 성공"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_product = Product(
            product_id=1, barcode='001', name='사과', quantity=50, price=1000,
            section_id=1, category='fruit', allergy_info_id=1, is_vegan_friendly=True
        )
        mock_session.query.return_value.filter_by.return_value.with_for_update.return_value.first.return_value = mock_product

        # Act
        result = await inventory_service.check_and_reserve_stock(1, 10)

        # Assert
        assert result is True
        assert mock_product.quantity == 40  # 50 - 10
        mock_session.commit.assert_called_once()

    async def test_check_and_reserve_stock_insufficient(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """재고 부족"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_product = Product(
            product_id=1, barcode='001', name='사과', quantity=5, price=1000,
            section_id=1, category='fruit', allergy_info_id=1, is_vegan_friendly=True
        )
        mock_session.query.return_value.filter_by.return_value.with_for_update.return_value.first.return_value = mock_product

        # Act
        result = await inventory_service.check_and_reserve_stock(1, 10)

        # Assert
        assert result is False
        assert mock_product.quantity == 5  # 변경 없음
        mock_session.commit.assert_not_called()

    async def test_check_and_reserve_stock_product_not_found(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """존재하지 않는 상품 예약"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.with_for_update.return_value.first.return_value = None

        # Act
        result = await inventory_service.check_and_reserve_stock(999, 10)

        # Assert
        assert result is False

    async def test_release_stock_success(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """재고 해제 성공"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_product = Product(
            product_id=1, barcode='001', name='사과', quantity=40, price=1000,
            section_id=1, category='fruit', allergy_info_id=1, is_vegan_friendly=True
        )
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_product

        # Act
        await inventory_service.release_stock(1, 10)

        # Assert
        assert mock_product.quantity == 50  # 40 + 10
        mock_session.commit.assert_called_once()

    async def test_release_stock_product_not_found(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """존재하지 않는 상품 재고 해제 (경고만 로그)"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Act (예외 발생하지 않음)
        await inventory_service.release_stock(999, 10)

        # Assert
        mock_session.commit.assert_not_called()


class TestInventoryServiceGetPose:
    """좌표 조회 테스트"""

    async def test_get_location_pose_success(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """Location 좌표 조회 성공"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_location = Location(location_id=1, location_x=10.5, location_y=20.3, location_theta=1.57)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_location

        # Act
        pose = await inventory_service.get_location_pose(1)

        # Assert
        assert pose is not None
        assert pose['x'] == 10.5
        assert pose['y'] == 20.3
        assert pose['theta'] == 1.57

    async def test_get_location_pose_not_found(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """Location 없음"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Act
        pose = await inventory_service.get_location_pose(999)

        # Assert
        assert pose is None

    async def test_get_warehouse_pose_success(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """Warehouse 좌표 조회 성공"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_location = Location(location_id=1, location_x=5.0, location_y=10.0, location_theta=0.0)
        mock_warehouse = Warehouse(warehouse_id=1, location_id=1)
        mock_warehouse.location = mock_location
        mock_session.query.return_value.filter.return_value.first.return_value = mock_warehouse

        # Act
        pose = await inventory_service.get_warehouse_pose(1)

        # Assert
        assert pose is not None
        assert pose['x'] == 5.0
        assert pose['y'] == 10.0

    async def test_get_section_pose_success(
        self, inventory_service: InventoryService, mock_db_manager: MagicMock
    ):
        """Section 좌표 조회 성공"""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_location = Location(location_id=1, location_x=3.0, location_y=7.0, location_theta=0.5)
        mock_section = Section(section_id=1)
        mock_section.location = mock_location
        mock_session.query.return_value.filter.return_value.first.return_value = mock_section

        # Act
        pose = await inventory_service.get_section_pose(1)

        # Assert
        assert pose is not None
        assert pose['x'] == 3.0
        assert pose['y'] == 7.0
        assert pose['theta'] == 0.5
