
#!/usr/bin/env python3
"""
Shopee Main Service - Database State Verification Test

이 스크립트는 `test_client.py`의 전체 워크플로우 테스트 실행 후,
데이터베이스의 상태가 예상대로 변경되었는지 검증합니다.

주요 검증 항목:
1. 주문(Order) 및 주문 항목(OrderItem)이 정상적으로 생성되었는가?
2. 주문 완료 후 상품(Product)의 재고가 올바르게 차감되었는가?
3. 로봇의 작업 내역(RobotHistory)이 기록되었는가?

사용 방법:
1. Mock LLM, Robot, Main Service를 모두 실행
2. 이 스크립트를 실행: python3 scripts/verify_db_state.py
3. 스크립트가 자동으로 test_client.py를 실행하고 DB 상태를 검증

주의: 
- 데이터베이스가 설정되어 있어야 합니다 (.env 파일 필요)
- Mock 환경이 실행 중이어야 합니다
"""

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from sqlalchemy.sql import func

# --- 경로 설정 ---
SCRIPT_DIR = Path(__file__).parent.absolute()
PACKAGE_DIR = SCRIPT_DIR.parent  # shopee_main_service 디렉토리
ENV_FILE = PACKAGE_DIR / ".env"

# --- 환경 변수 및 설정 ---
def get_db_url_from_env():
    """
    .env 파일에서 데이터베이스 URL을 읽어옵니다.
    """
    if not ENV_FILE.exists():
        print(f"Error: .env file not found at {ENV_FILE}")
        print("Please create it from .env.example")
        sys.exit(1)
    
    try:
        with open(ENV_FILE) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        if key == "SHOPEE_DB_URL":
                            print(f"Using database URL from .env: {value.split('@')[-1]}")
                            return value
    except Exception as e:
        print(f"Error reading .env file: {e}")
        sys.exit(1)
    
    # Fallback
    print("Warning: SHOPEE_DB_URL not found in .env, using default")
    return "mysql+pymysql://shopee:shopee@localhost:3306/shopee"

DB_URL = get_db_url_from_env()
TEST_USER_ID = "admin"
TEST_PRODUCT_ID = 1  # 테스트할 상품 ID
TEST_PRODUCT_QUANTITY = 2  # 주문할 상품 수량

# --- SQLAlchemy 모델 (database_models.py와 일치) ---
Base = declarative_base()

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

class Section(Base):
    __tablename__ = 'section'
    section_id = Column(Integer, primary_key=True, autoincrement=True)
    shelf_id = Column(Integer, ForeignKey('shelf.shelf_id'), nullable=False)
    section_name = Column(String(50), nullable=False)

class Shelf(Base):
    __tablename__ = 'shelf'
    shelf_id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=False)
    shelf_name = Column(String(50), nullable=False)

class Warehouse(Base):
    __tablename__ = 'warehouse'
    warehouse_id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=False)
    warehouse_name = Column(String(50), nullable=False)

class Order(Base):
    __tablename__ = 'order'
    order_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey('customer.customer_id'), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    order_status = Column(mysql.TINYINT, nullable=False)
    failure_reason = Column(String(50), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = 'order_item'
    order_item_id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey('order.order_id'), nullable=False)
    product_id = Column(Integer, ForeignKey('product.product_id'), nullable=False)
    quantity = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    order = relationship("Order", back_populates="items")

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


# --- 데이터베이스 세션 관리 ---
engine = create_engine(DB_URL)
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# --- 테스트 실행 함수 ---
def run_workflow_test():
    """
    test_client.py를 서브프로세스로 실행
    
    주의: Mock LLM, Robot Node, Main Service가 이미 실행 중이어야 합니다.
    """
    print("\n" + "="*20 + " Running Workflow Test " + "="*20)
    print("Checking if Main Service is available on localhost:5000...")
    
    # Main Service가 실행 중인지 간단히 체크
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 5000))
        sock.close()
        if result != 0:
            print("\n!!! Main Service is not running on port 5000 !!!")
            print("Please start Mock LLM, Robot Node, and Main Service first.")
            print("\nRequired services:")
            print("  1. ros2 run shopee_main_service mock_llm_server")
            print("  2. ros2 run shopee_main_service mock_robot_node")
            print("  3. ros2 run shopee_main_service main_service_node")
            sys.exit(1)
    except Exception as e:
        print(f"Error checking Main Service: {e}")
        sys.exit(1)
    
    print("Main Service is running. Starting test...")
    
    try:
        script_path = SCRIPT_DIR / "test_client.py"
        process = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            cwd=str(PACKAGE_DIR)  # shopee_main_service 디렉토리에서 실행
        )
        print("Workflow test completed successfully.")
        # 필요시 출력 표시
        if process.stdout:
            print("\n--- Test Output (last 10 lines) ---")
            print('\n'.join(process.stdout.split('\n')[-10:]))
    except subprocess.CalledProcessError as e:
        print("\n!!! Workflow test script failed! !!!")
        print(e.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("\n!!! Workflow test timed out! !!!")
        sys.exit(1)
    print("="*63 + "\n")


# --- 메인 검증 로직 ---
def verify_database_state():
    print("="*25 + " DB State Verification " + "="*25)
    
    initial_quantity = -1
    
    # 1. 테스트 전 초기 상태 확인
    print("[Step 1] Checking initial state...")
    with session_scope() as session:
        test_product = session.query(Product).filter_by(product_id=TEST_PRODUCT_ID).first()
        if not test_product:
            print(f"Error: Test product with ID {TEST_PRODUCT_ID} not found in database.")
            sys.exit(1)
        initial_quantity = test_product.quantity
        
        test_customer = session.query(Customer).filter_by(id=TEST_USER_ID).first()
        if not test_customer:
            print(f"Error: Test customer with ID {TEST_USER_ID} not found.")
            sys.exit(1)

        # 이전 테스트에서 남은 주문 데이터 정리
        session.query(RobotHistory).delete()
        session.query(OrderItem).delete()
        session.query(Order).delete()
        print(f"Cleaned up previous test orders for user '{TEST_USER_ID}'.")

    print(f"Initial quantity of product '{test_product.name}' (ID: {TEST_PRODUCT_ID}): {initial_quantity}")
    
    # 2. 워크플로우 테스트 실행
    run_workflow_test()
    
    # 3. 테스트 후 최종 상태 검증
    print("[Step 2] Verifying final state in database...")
    
    report = {
        "order_creation": "FAIL",
        "order_status": "FAIL",
        "product_quantity": "FAIL",
        "robot_history": "FAIL",
    }
    
    with session_scope() as session:
        # 가장 최근 주문 조회
        latest_order = session.query(Order).order_by(Order.order_id.desc()).first()
        
        if latest_order:
            report["order_creation"] = "PASS"
            
            # 주문 상태 검증 (8: PACKED)
            if latest_order.order_status == 8:
                report["order_status"] = "PASS"
            else:
                report["order_status"] = f"FAIL (Expected: 8, Actual: {latest_order.order_status})"
                
            # 주문 항목 검증
            order_item = session.query(OrderItem).filter_by(order_id=latest_order.order_id, product_id=TEST_PRODUCT_ID).first()
            if order_item:
                # 재고 수량 검증
                final_product = session.query(Product).filter_by(product_id=TEST_PRODUCT_ID).first()
                expected_quantity = initial_quantity - order_item.quantity
                if final_product.quantity == expected_quantity:
                    report["product_quantity"] = "PASS"
                else:
                    report["product_quantity"] = f"FAIL (Expected: {expected_quantity}, Actual: {final_product.quantity})"
                
                # 로봇 기록 검증
                history_count = session.query(RobotHistory).filter_by(order_item_id=order_item.order_item_id).count()
                if history_count > 0:
                    report["robot_history"] = f"PASS ({history_count} records found)"
                else:
                    report["robot_history"] = "FAIL (No records found)"
            else:
                report["product_quantity"] = "FAIL (OrderItem not found)"
                report["robot_history"] = "FAIL (OrderItem not found)"

    # 4. 결과 리포트 출력
    print("\n" + "-"*28 + " REPORT " + "-"*27)
    for key, value in report.items():
        status = "PASS" if "PASS" in value else "FAIL"
        print(f"- {key:<20}: {value}")
    print("-" * 63)

    if "FAIL" in str(report.values()):
        print("\n[!] Verification failed. Bugs detected in data persistence logic.")
    else:
        print("\n[+] All database verifications passed!")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║   Shopee Main Service - DB State Verification            ║
╚═══════════════════════════════════════════════════════════╝

이 스크립트는:
1. test_client.py를 자동으로 실행
2. 데이터베이스 상태를 검증
3. 주문, 재고, 로봇 히스토리가 올바른지 확인

필수 조건:
- Mock LLM Server가 실행 중이어야 함
- Mock Robot Node가 실행 중이어야 함
- Main Service가 실행 중이어야 함
- 데이터베이스가 설정되어 있어야 함
""")
    
    try:
        verify_database_state()
    except KeyboardInterrupt:
        print("\n\n[!] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
