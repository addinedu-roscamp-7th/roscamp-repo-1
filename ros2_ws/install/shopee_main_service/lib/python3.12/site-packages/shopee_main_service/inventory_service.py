"""
재고 관리 서비스

상품 검색/추가/수정/삭제 기능을 제공합니다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_

from .database_models import Product, Section, Warehouse


class InventoryService:
    """재고(상품) 관련 비즈니스 로직"""

    def __init__(self, db):
        self._db = db

    async def search_products(self, filters: Dict[str, object]) -> Tuple[List[Dict[str, object]], int]:
        """
        상품 검색

        Args:
            filters: 검색 조건

        Returns:
            (상품 목록, 총 개수)
        """
        with self._db.session_scope() as session:
            query = session.query(Product)

            if (product_id := filters.get("product_id")) is not None:
                query = query.filter(Product.product_id == product_id)
            if (barcode := filters.get("barcode")):
                query = query.filter(Product.barcode == barcode)
            if (name := filters.get("name")):
                query = query.filter(Product.name.like(f"%{name}%"))
            if (price := filters.get("price")) is not None:
                query = query.filter(Product.price == price)
            if (section_id := filters.get("section_id")) is not None:
                query = query.filter(Product.section_id == section_id)
            if (category := filters.get("category")):
                query = query.filter(Product.category == category)
            if (allergy_info_id := filters.get("allergy_info_id")) is not None:
                query = query.filter(Product.allergy_info_id == allergy_info_id)
            if (is_vegan := filters.get("is_vegan_friendly")) is not None:
                query = query.filter(Product.is_vegan_friendly == self._to_bool(is_vegan))

            quantity = filters.get("quantity")
            if isinstance(quantity, (list, tuple)) and quantity:
                conditions = []
                if quantity[0] is not None:
                    conditions.append(Product.quantity >= int(quantity[0]))
                if len(quantity) > 1 and quantity[1] is not None:
                    conditions.append(Product.quantity <= int(quantity[1]))
                if conditions:
                    query = query.filter(and_(*conditions))

            products = query.all()
            product_dicts = [self._product_to_dict(product) for product in products]
            return product_dicts, len(product_dicts)

    async def create_product(self, payload: Dict[str, object]) -> None:
        """상품 추가"""
        required_fields = [
            "product_id", "barcode", "name", "quantity", "price",
            "section_id", "category", "allergy_info_id", "is_vegan_friendly"
        ]
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")

        with self._db.session_scope() as session:
            existing = session.query(Product).filter_by(product_id=payload["product_id"]).first()
            if existing:
                raise ValueError("Product already exists.")

            warehouse_id = self._resolve_warehouse_id(session, payload["section_id"])

            product = Product(
                product_id=int(payload["product_id"]),
                barcode=str(payload["barcode"]),
                name=str(payload["name"]),
                quantity=int(payload["quantity"]),
                price=int(payload["price"]),
                discount_rate=int(payload.get("discount_rate", 0)),
                category=str(payload["category"]),
                allergy_info_id=int(payload["allergy_info_id"]),
                is_vegan_friendly=self._to_bool(payload["is_vegan_friendly"]),
                section_id=int(payload["section_id"]),
                warehouse_id=warehouse_id,
            )
            session.add(product)

    async def update_product(self, payload: Dict[str, object]) -> bool:
        """상품 수정"""
        product_id = payload.get("product_id")
        if product_id is None:
            raise ValueError("product_id is required.")

        with self._db.session_scope() as session:
            product = session.query(Product).filter_by(product_id=product_id).first()
            if not product:
                return False

            for field in ("barcode", "name", "category"):
                if field in payload and payload[field] is not None:
                    setattr(product, field, str(payload[field]))

            if "quantity" in payload and payload["quantity"] is not None:
                product.quantity = int(payload["quantity"])

            if "price" in payload and payload["price"] is not None:
                product.price = int(payload["price"])

            if "allergy_info_id" in payload and payload["allergy_info_id"] is not None:
                product.allergy_info_id = int(payload["allergy_info_id"])

            if "is_vegan_friendly" in payload and payload["is_vegan_friendly"] is not None:
                product.is_vegan_friendly = self._to_bool(payload["is_vegan_friendly"])

            if "section_id" in payload and payload["section_id"] is not None:
                product.section_id = int(payload["section_id"])
                product.warehouse_id = self._resolve_warehouse_id(session, product.section_id, fallback=product.warehouse_id)

            if "discount_rate" in payload and payload["discount_rate"] is not None:
                product.discount_rate = int(payload["discount_rate"])

            return True

    async def delete_product(self, product_id: int) -> bool:
        """상품 삭제"""
        with self._db.session_scope() as session:
            product = session.query(Product).filter_by(product_id=product_id).first()
            if not product:
                return False
            session.delete(product)
            return True

    def _product_to_dict(self, product: Product) -> Dict[str, object]:
        return {
            "product_id": product.product_id,
            "barcode": product.barcode,
            "name": product.name,
            "quantity": product.quantity,
            "price": product.price,
            "section_id": product.section_id,
            "category": product.category,
            "allergy_info_id": product.allergy_info_id,
            "is_vegan_friendly": product.is_vegan_friendly,
        }

    def _resolve_warehouse_id(self, session, section_id: int, fallback: Optional[int] = None) -> int:
        """Section 정보를 바탕으로 Warehouse ID 추론"""
        if fallback is not None:
            return fallback

        if section_id is None:
            return 1

        section = session.query(Section).filter_by(section_id=section_id).first()
        if section and section.shelf:
            warehouse = session.query(Warehouse).filter_by(location_id=section.shelf.location_id).first()
            if warehouse:
                return warehouse.warehouse_id
        return 1

    def _to_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "y"}
        return bool(value)
