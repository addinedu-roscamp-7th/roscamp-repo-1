"""
SQLAlchemy ORM 모델 정의

데이터베이스의 각 테이블에 매핑되는 파이썬 클래스를 정의합니다.
- `Base`를 상속받아 모든 모델을 관리합니다.
- 각 클래스는 하나의 테이블을 나타냅니다.
- 클래스 속성은 테이블의 컬럼에 해당합니다.
- `relationship`을 통해 테이블 간의 관계(FK)를 정의합니다.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, Float, DateTime, ForeignKey
)
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

# 모든 ORM 모델이 상속받을 기본 클래스
Base = declarative_base()

# --- 테이블 모델 정의 --- #

class AllergyInfo(Base):
    __tablename__ = 'allergy_info'
    allergy_info_id = Column(Integer, primary_key=True, autoincrement=True)
    nuts = Column(Boolean, nullable=False)
    milk = Column(Boolean, nullable=False)
    seafood = Column(Boolean, nullable=False)
    soy = Column(Boolean, nullable=False)
    peach = Column(Boolean, nullable=False)
    gluten = Column(Boolean, nullable=False)
    eggs = Column(Boolean, nullable=False)

class Admin(Base):
    __tablename__ = 'admin'
    admin_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String(20), nullable=False, unique=True)
    password = Column(String(256), nullable=False)
    name = Column(String(10), nullable=False)

class Customer(Base):
    __tablename__ = 'customer'
    customer_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String(20), nullable=False, unique=True)
    password = Column(String(256), nullable=False)
    name = Column(String(10), nullable=False)
    gender = Column(Boolean, nullable=False)
    age = Column(Integer, nullable=False)
    address = Column(String(256), nullable=False)
    allergy_info_id = Column(Integer, ForeignKey('allergy_info.allergy_info_id'), nullable=False)
    is_vegan = Column(Boolean, nullable=False)

    allergy_info = relationship("AllergyInfo")

class Location(Base):
    __tablename__ = 'location'
    location_id = Column(Integer, primary_key=True, autoincrement=True)
    location_x = Column(Float, nullable=False)
    location_y = Column(Float, nullable=False)
    location_theta = Column(Float, nullable=False, default=0.0)
    aruco_marker = Column(Integer, nullable=False)

class Warehouse(Base):
    __tablename__ = 'warehouse'
    warehouse_id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=False)
    warehouse_name = Column(String(50), nullable=False)

    location = relationship("Location")

class Shelf(Base):
    __tablename__ = 'shelf'
    shelf_id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=False)
    shelf_name = Column(String(50), nullable=False)

    location = relationship("Location")

class Section(Base):
    __tablename__ = 'section'
    section_id = Column(Integer, primary_key=True, autoincrement=True)
    shelf_id = Column(Integer, ForeignKey('shelf.shelf_id'), nullable=False)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=False)
    section_name = Column(String(50), nullable=False)

    shelf = relationship("Shelf")
    location = relationship("Location")

class Product(Base):
    __tablename__ = 'product'
    product_id = Column(Integer, primary_key=True, autoincrement=True)
    barcode = Column(String(100), nullable=False, unique=True)
    name = Column(String(50), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    discount_rate = Column(Integer, nullable=False)
    category = Column(String(10), nullable=False)
    allergy_info_id = Column(Integer, ForeignKey('allergy_info.allergy_info_id'), nullable=False)
    is_vegan_friendly = Column(Boolean, nullable=False)
    section_id = Column(Integer, ForeignKey('section.section_id'), nullable=False)
    warehouse_id = Column(Integer, ForeignKey('warehouse.warehouse_id'), nullable=False)
    length = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    weight = Column(Integer, nullable=True)
    fragile = Column(Boolean, nullable=True)
    img_path = Column(String(50), nullable=True)

    allergy_info = relationship("AllergyInfo")
    section = relationship("Section")
    warehouse = relationship("Warehouse")

class Box(Base):
    __tablename__ = 'box'
    box_id = Column(Integer, primary_key=True, autoincrement=True)
    width = Column(Integer, nullable=False)
    depth = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)


class Order(Base):
    __tablename__ = 'order'
    order_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey('customer.customer_id'), nullable=False)
    box_id = Column(Integer, ForeignKey('box.box_id'), nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    order_status = Column(mysql.TINYINT, nullable=False)
    failure_reason = Column(String(50), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    customer = relationship("Customer")
    items = relationship("OrderItem", back_populates="order")
    box = relationship("Box")

class OrderItem(Base):
    __tablename__ = 'order_item'
    order_item_id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey('order.order_id'), nullable=False)
    product_id = Column(Integer, ForeignKey('product.product_id'), nullable=False)
    quantity = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    order = relationship("Order", back_populates="items")
    product = relationship("Product")

class Robot(Base):
    __tablename__ = 'robot'
    robot_id = Column(Integer, primary_key=True, autoincrement=True)
    robot_type = Column(mysql.TINYINT, nullable=False)
    robot_status = Column(mysql.TINYINT, nullable=False)

class RobotHistory(Base):
    __tablename__ = 'robot_history'
    robot_history_id = Column(Integer, primary_key=True, autoincrement=True)
    robot_id = Column(Integer, ForeignKey('robot.robot_id'), nullable=False)
    history_type = Column(mysql.TINYINT, nullable=False)
    order_item_id = Column(Integer, ForeignKey('order_item.order_item_id'), nullable=False)
    is_complete = Column(Boolean, nullable=False)
    failure_reason = Column(String(50), nullable=True)
    active_duration = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    robot = relationship("Robot")
    order_item = relationship("OrderItem")
