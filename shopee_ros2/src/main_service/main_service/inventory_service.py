"""
재고 관리 서비스

상품 검색/추가/수정/삭제 기능을 제공합니다.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_
from sqlalchemy.orm import Query, load_only

from .database_models import Product, Section, Warehouse, Location

logger = logging.getLogger(__name__)


class InventoryService:
    """재고(상품) 관련 비즈니스 로직"""

    def __init__(self, db):
        self._db = db

    async def get_location_pose(self, location_id: int) -> Optional[Dict[str, float]]:
        """
        Location ID에 해당하는 좌표(Pose)를 조회합니다.

        Args:
            location_id: 조회할 위치의 ID

        Returns:
            좌표 정보를 담은 딕셔너리 또는 None
        """
        return self.get_location_pose_sync(location_id)

    def get_location_pose_sync(self, location_id: int) -> Optional[Dict[str, float]]:
        """동기 방식 좌표 조회"""
        with self._db.session_scope() as session:
            location = session.query(Location).filter(Location.location_id == location_id).first()
            if not location:
                return None
            return {
                "x": location.location_x,
                "y": location.location_y,
                "theta": location.location_theta,
            }

    async def get_warehouse_pose(self, warehouse_id: int) -> Optional[Dict[str, float]]:
        """
        Warehouse ID에 해당하는 좌표(Pose)를 조회합니다.

        Args:
            warehouse_id: 조회할 창고의 ID

        Returns:
            좌표 정보를 담은 딕셔너리 또는 None
        """
        return self.get_warehouse_pose_sync(warehouse_id)

    def get_warehouse_pose_sync(self, warehouse_id: int) -> Optional[Dict[str, float]]:
        """동기 방식 창고 좌표 조회"""
        with self._db.session_scope() as session:
            warehouse = (
                session.query(Warehouse)
                .filter(Warehouse.warehouse_id == warehouse_id)
                .first()
            )
            if warehouse and warehouse.location:
                loc = warehouse.location
                return {
                    "x": loc.location_x,
                    "y": loc.location_y,
                    "theta": loc.location_theta,
                }
        return None

    async def get_section_pose(self, section_id: int) -> Optional[Dict[str, float]]:
        """
        Section ID에 해당하는 좌표(Pose)를 조회합니다.

        Args:
            section_id: 조회할 섹션의 ID

        Returns:
            좌표 정보를 담은 딕셔너리 또는 None
        """
        return self.get_section_pose_sync(section_id)

    def get_section_pose_sync(self, section_id: int) -> Optional[Dict[str, float]]:
        """동기 방식 섹션 좌표 조회"""
        with self._db.session_scope() as session:
            section = (
                session.query(Section)
                .filter(Section.section_id == section_id)
                .first()
            )
            if section and section.location:
                loc = section.location
                return {
                    "x": loc.location_x,
                    "y": loc.location_y,
                    "theta": loc.location_theta,
                }
        return None

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
                length=int(payload["length"]) if payload.get("length") is not None else None,
                width=int(payload["width"]) if payload.get("width") is not None else None,
                height=int(payload["height"]) if payload.get("height") is not None else None,
                weight=int(payload["weight"]) if payload.get("weight") is not None else None,
                fragile=self._to_bool(payload["fragile"]) if payload.get("fragile") is not None else None,
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

            for field in ("quantity", "price", "allergy_info_id", "discount_rate", "length", "width", "height", "weight"):
                if field in payload and payload[field] is not None:
                    setattr(product, field, int(payload[field]))

            if "is_vegan_friendly" in payload and payload["is_vegan_friendly"] is not None:
                product.is_vegan_friendly = self._to_bool(payload["is_vegan_friendly"])
            
            if "fragile" in payload and payload["fragile"] is not None:
                product.fragile = self._to_bool(payload["fragile"])

            if "section_id" in payload and payload["section_id"] is not None:
                product.section_id = int(payload["section_id"])
                product.warehouse_id = self._resolve_warehouse_id(session, product.section_id, fallback=product.warehouse_id)

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
            "length": product.length,
            "width": product.width,
            "height": product.height,
            "weight": product.weight,
            "fragile": product.fragile,
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

    async def check_and_reserve_stock(
        self,
        product_id: int,
        quantity: int
    ) -> bool:
        """
        재고 확인 및 차감

        Args:
            product_id: 상품 ID
            quantity: 차감할 수량

        Returns:
            True: 재고 충분하여 차감 성공
            False: 재고 부족
        """
        with self._db.session_scope() as session:
            query = session.query(Product)
            if isinstance(query, Query):
                query = query.options(load_only(Product.quantity))

            # SELECT FOR UPDATE로 락 걸기 (동시성 제어)
            product = query.filter_by(
                product_id=product_id
            ).with_for_update().first()

            if not product:
                logger.warning(f"No product found for product_id={product_id}")
                return False

            if product.quantity < quantity:
                logger.warning(
                    f"Insufficient stock for product {product_id}: "
                    f"requested={quantity}, available={product.quantity}"
                )
                return False

            # 재고 차감
            product.quantity -= quantity
            session.commit()
            logger.info(
                f"Reserved stock: product={product_id}, qty={quantity}, "
                f"remaining={product.quantity}"
            )
            return True

    async def release_stock(
        self,
        product_id: int,
        quantity: int
    ) -> None:
        """
        주문 취소 시 재고 복구

        Args:
            product_id: 상품 ID
            quantity: 복구할 수량
        """
        with self._db.session_scope() as session:
            query = session.query(Product)
            if isinstance(query, Query):
                query = query.options(load_only(Product.quantity))

            product = query.filter_by(
                product_id=product_id
            ).first()

            if product:
                product.quantity += quantity
                session.commit()
                logger.info(
                    f"Released stock: product={product_id}, qty={quantity}, "
                    f"total={product.quantity}"
                )
            else:
                logger.warning(
                    f"Cannot release stock: product {product_id} not found"
                )
