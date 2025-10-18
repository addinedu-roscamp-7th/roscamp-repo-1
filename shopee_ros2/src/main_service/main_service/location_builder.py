"""
상품 위치 정보 빌더

주문의 상품 위치 정보를 생성합니다.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from shopee_interfaces.msg import ProductLocation

if TYPE_CHECKING:
    from .database_manager import DatabaseManager
    from .database_models import OrderItem

logger = logging.getLogger(__name__)


class ProductLocationBuilder:
    """
    주문 항목을 기반으로 ProductLocation 리스트를 생성합니다.
    """

    def __init__(self, db: "DatabaseManager") -> None:
        self._db = db

    def build_product_locations(self, session, order_id: int) -> List[ProductLocation]:
        """주문 항목을 기반으로 ProductLocation 리스트 생성"""
        from .database_models import OrderItem, Product
        
        locations: List[ProductLocation] = []
        items = session.query(OrderItem).filter_by(order_id=order_id).all()
        
        for item in items:
            product = (
                session.query(Product)
                .filter_by(product_id=item.product_id)
                .first()
            )
            if not product or not product.section or not product.section.shelf:
                logger.warning(
                    'Skipping product %s for order %s due to missing section/shelf mapping',
                    item.product_id,
                    order_id,
                )
                continue
            
            locations.append(
                ProductLocation(
                    product_id=product.product_id,
                    location_id=product.section.shelf.location_id,
                    section_id=product.section_id,
                    quantity=item.quantity,
                )
            )
        
        return locations

