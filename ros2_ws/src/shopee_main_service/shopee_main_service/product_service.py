"""
상품 검색 및 재고 관리 서비스

LLM 기반 자연어 검색과 재고 관리 기능을 제공합니다.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import text

from .database_models import Product

if TYPE_CHECKING:
    from .database_manager import DatabaseManager
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProductService:
    """
    상품 서비스
    
    - LLM 기반 자연어 검색
    - 재고 조회 및 업데이트
    - 알레르기/비건 필터링
    """

    def __init__(self, db: "DatabaseManager", llm_client: "LLMClient") -> None:
        self._db = db
        self._llm = llm_client

    async def search_products(self, query: str, filters: Optional[dict] = None) -> Dict[str, Any]:
        """
        상품 검색 (LLM 연동)

        Args:
            query: 자연어 검색어 (예: "비건 사과")
            filters: 추가 필터 (현재 미사용)

        Returns:
            dict: {"products": list[dict], "total_count": int}
        """
        # 1. LLM을 통해 검색 조건 생성
        # 예: "name LIKE '%사과%' AND is_vegan_friendly = true"
        where_clause = await self._llm.generate_search_query(query)

        # 2. LLM 호출 실패 또는 결과가 없을 경우, 기본 검색 로직 수행 (Fallback)
        if not where_clause:
            logger.warning("LLM query generation failed. Falling back to basic search.")
            # 간단한 LIKE 검색으로 대체
            keyword = f"%{query}%"
            with self._db.session_scope() as session:
                products = session.query(Product).filter(Product.name.like(keyword)).all()
                # 세션 안에서 딕셔너리로 변환 (DetachedInstanceError 방지)
                product_list = [self._product_to_dict(p) for p in products]
                return {
                    "products": product_list,
                    "total_count": len(product_list)
                }
        else:
            # 3. LLM이 생성한 WHERE 절을 사용하여 안전하게 쿼리 실행
            # SQLAlchemy의 text()를 사용하여 SQL Injection 방지
            # SELECT * 와 같은 위험한 구문 대신, 모델(Product)을 기반으로 조회
            full_query = text(f"SELECT * FROM product WHERE {where_clause}")
            with self._db.session_scope() as session:
                products = session.query(Product).from_statement(full_query).all()
                # 세션 안에서 딕셔너리로 변환 (DetachedInstanceError 방지)
                product_list = [self._product_to_dict(p) for p in products]
                return {
                    "products": product_list,
                    "total_count": len(product_list)
                }

    def _product_to_dict(self, product: Product) -> Dict[str, Any]:
        """Product 객체를 딕셔너리로 변환"""
        return {
            "product_id": product.product_id,
            "barcode": product.barcode,
            "name": product.name,
            "quantity": product.quantity,
            "price": product.price,
            "discount_rate": product.discount_rate,
            "category": product.category,
            "is_vegan_friendly": product.is_vegan_friendly,
            "allergy_info_id": product.allergy_info_id, # 상세 정보는 필요 시 추가 조회
            "section_id": product.section_id,
            "warehouse_id": product.warehouse_id,
        }

    async def update_inventory(self, product_id: int, quantity_change: int) -> bool:
        """
        재고 업데이트
        
        Args:
            product_id: 상품 ID
            quantity_change: 수량 변경 (양수: 입고, 음수: 출고)
            
        Returns:
            bool: 성공 여부
        """
        with self._db.session_scope() as session:
            product = session.query(Product).filter_by(product_id=product_id).first()
            if product:
                product.quantity += quantity_change
                logger.info(
                    "Inventory updated for product %d. New quantity: %d",
                    product_id, product.quantity
                )
                return True
        logger.warning("Inventory update failed: Product %d not found", product_id)
        return False

    def get_product_location_sync(self, product_id: int) -> Optional[Dict[str, int]]:
        """
        상품 ID로 위치 정보(창고, 섹션)를 동기적으로 조회합니다.
        ROS2 서비스 콜백에서 사용하기 위해 동기 메서드로 구현합니다.
        
        Args:
            product_id: 상품 ID
            
        Returns:
            {"warehouse_id": int, "section_id": int} 딕셔너리 또는 None
        """
        with self._db.session_scope() as session:
            product = session.query(Product).filter_by(product_id=product_id).first()
            if product:
                return {
                    "warehouse_id": product.warehouse_id,
                    "section_id": product.section_id,
                }
        return None
