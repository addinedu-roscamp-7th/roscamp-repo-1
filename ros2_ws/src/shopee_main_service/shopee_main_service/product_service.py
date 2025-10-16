"""
상품 검색 및 재고 관리 서비스

LLM 기반 자연어 검색과 재고 관리 기능을 제공합니다.
"""
from __future__ import annotations

import logging
import re
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
        self._allowed_columns = {
            "product_id",
            "barcode",
            "name",
            "quantity",
            "price",
            "discount_rate",
            "category",
            "is_vegan_friendly",
            "allergy_info_id",
            "section_id",
            "warehouse_id",
        }
        self._allowed_keywords = {
            "AND",
            "OR",
            "NOT",
            "LIKE",
            "IN",
            "BETWEEN",
            "IS",
            "NULL",
            "TRUE",
            "FALSE",
            "ASC",
            "DESC",
            "LOWER",
            "UPPER",
        }
        self._disallowed_tokens = {
            " drop ",
            " insert ",
            " update ",
            " delete ",
            " union ",
            " select ",
            " alter ",
            " create ",
            ";",
            "--",
            "/*",
            "*/",
        }
        self._identifier_pattern = re.compile(r"\b([a-z_][a-z0-9_]*)\b", re.IGNORECASE)

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
            return self._basic_keyword_search(query)

        if not self._is_safe_where_clause(where_clause):
            logger.warning("Discarding unsafe LLM WHERE clause: %s", where_clause)
            return self._basic_keyword_search(query)

        # 3. LLM이 생성한 WHERE 절을 사용하여 안전하게 쿼리 실행
        full_query = text(f"SELECT * FROM product WHERE {where_clause}")
        with self._db.session_scope() as session:
            products = session.query(Product).from_statement(full_query).all()
            product_list = [self._product_to_dict(p) for p in products]
            return {
                "products": product_list,
                "total_count": len(product_list)
            }

    def _basic_keyword_search(self, keyword: str) -> Dict[str, Any]:
        """LLM 실패 시 기본 LIKE 검색을 수행합니다."""
        pattern = f"%{keyword}%"
        with self._db.session_scope() as session:
            products = session.query(Product).filter(Product.name.like(pattern)).all()
            product_list = [self._product_to_dict(p) for p in products]
            return {
                "products": product_list,
                "total_count": len(product_list)
            }

    def _is_safe_where_clause(self, clause: str) -> bool:
        """LLM이 생성한 WHERE 절이 허용된 키워드/컬럼만 포함하는지 검증합니다."""
        lowered = clause.lower()
        for token in self._disallowed_tokens:
            if token in lowered:
                return False

        for match in self._identifier_pattern.finditer(clause):
            token = match.group(1)
            if token.isdigit():
                continue
            lower = token.lower()
            if lower in self._allowed_columns:
                continue
            if token.upper() in self._allowed_keywords:
                continue
            if lower in {"true", "false", "null"}:
                continue
            return False
        return True

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

    async def get_product_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        상품명을 이용해 단일 상품 정보를 조회합니다.
        """
        if not name:
            return None

        with self._db.session_scope() as session:
            product = (
                session.query(Product)
                .filter(Product.name == name)
                .first()
            )
            if product:
                return self._product_to_dict(product)
        return None

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
