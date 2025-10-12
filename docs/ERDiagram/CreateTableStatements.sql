SET FOREIGN_KEY_CHECKS=0;

-- Drop existing tables
DROP TABLE IF EXISTS `robot_history`;
DROP TABLE IF EXISTS `order_item`;
DROP TABLE IF EXISTS `order`;
DROP TABLE IF EXISTS `product`;
DROP TABLE IF EXISTS `section`;
DROP TABLE IF EXISTS `shelf`;
DROP TABLE IF EXISTS `warehouse`;
DROP TABLE IF EXISTS `location`;
DROP TABLE IF EXISTS `customer`;
DROP TABLE IF EXISTS `admin`;
DROP TABLE IF EXISTS `allergy_info`;
DROP TABLE IF EXISTS `robot`;

SET FOREIGN_KEY_CHECKS=1;

-- Shopee Database Schema - CREATE TABLE Statements
-- Generated: 2025-10-10

-- ================================
-- 1. allergy_info (먼저 생성 - FK 참조됨)
-- ================================
CREATE TABLE allergy_info (
    allergy_info_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '알레르기 정보 pk',
    nuts BOOLEAN NOT NULL COMMENT '견과류 알레르기 여부 (1. 대상, 0. 비대상)',
    milk BOOLEAN NOT NULL COMMENT '유제품 알레르기 여부 (1. 대상, 0. 비대상)',
    seafood BOOLEAN NOT NULL COMMENT '어패류 알레르기 여부 (1. 대상, 0. 비대상)',
    soy BOOLEAN NOT NULL COMMENT '대두/콩 알레르기 여부 (1. 대상, 0. 비대상)',
    peach BOOLEAN NOT NULL COMMENT '복숭아 알레르기 여부 (1. 대상, 0. 비대상)',
    gluten BOOLEAN NOT NULL COMMENT '밀, 글루텐 알레르기 여부 (1. 대상, 0. 비대상)',
    eggs BOOLEAN NOT NULL COMMENT '계란 알레르기 여부 (1. 대상, 0. 비대상)'
);

-- ================================
-- 2. admin
-- ================================
CREATE TABLE admin (
    admin_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '관리자 pkey',
    id VARCHAR(20) NOT NULL COMMENT '아이디',
    password VARCHAR(256) NOT NULL COMMENT '비밀번호',
    name VARCHAR(10) NOT NULL COMMENT '관리자명'
);

-- ================================
-- 3. customer (수정됨 - id, password 추가)
-- ================================
CREATE TABLE customer (
    customer_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '고객 정보 pkey',
    id VARCHAR(20) NOT NULL COMMENT '아이디',
    password VARCHAR(256) NOT NULL COMMENT '비밀번호',
    name VARCHAR(10) NOT NULL COMMENT '고객명',
    gender BOOLEAN NOT NULL COMMENT '성별 (1. 남성, 0. 여성)',
    age INT NOT NULL COMMENT '나이',
    address VARCHAR(256) NOT NULL COMMENT '주소',
    allergy_info_id INT NOT NULL COMMENT '알레르기 정보',
    is_vegan BOOLEAN NOT NULL COMMENT '비건 유무 (1. 대상, 0. 비대상)',
    FOREIGN KEY (allergy_info_id) REFERENCES allergy_info(allergy_info_id)
);

-- ================================
-- 4. location (FK 참조됨)
-- ================================
CREATE TABLE location (
    location_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '위치 정보 pkey',
    location_x FLOAT NOT NULL COMMENT '위치 x 좌표',
    location_y FLOAT NOT NULL COMMENT '위치 y 좌표',
    aruco_marker INT NOT NULL COMMENT '공간 번호'
);

-- ================================
-- 5. warehouse
-- ================================
CREATE TABLE warehouse (
    warehouse_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '창고 정보 pkey',
    location_id INT NOT NULL COMMENT '위치 정보',
    warehouse_name VARCHAR(50) NOT NULL COMMENT '창고 이름',
    FOREIGN KEY (location_id) REFERENCES location(location_id)
);

-- ================================
-- 6. shelf
-- ================================
CREATE TABLE shelf (
    shelf_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '매대 정보 pkey',
    location_id INT NOT NULL COMMENT '위치 정보',
    shelf_name VARCHAR(50) NOT NULL COMMENT '매대 이름',
    FOREIGN KEY (location_id) REFERENCES location(location_id)
);

-- ================================
-- 7. section
-- ================================
CREATE TABLE section (
    section_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '섹션 정보 pkey',
    shelf_id INT NOT NULL COMMENT '매대 정보',
    location_id INT NOT NULL COMMENT '위치 정보',
    section_name VARCHAR(50) NOT NULL COMMENT '코너 이름',
    FOREIGN KEY (shelf_id) REFERENCES shelf(shelf_id),
    FOREIGN KEY (location_id) REFERENCES location(location_id)
);

-- ================================
-- 8. product (수정됨 - name VARCHAR(50))
-- ================================
CREATE TABLE product (
    product_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '상품 정보 pkey',
    barcode VARCHAR(100) NOT NULL COMMENT '바코드',
    name VARCHAR(50) NOT NULL COMMENT '상품명',
    quantity INT NOT NULL COMMENT '수량',
    price INT NOT NULL COMMENT '가격',
    discount_rate INT NOT NULL COMMENT '할인율',
    category VARCHAR(10) NOT NULL COMMENT '구분',
    allergy_info_id INT NOT NULL COMMENT '알레르기 정보',
    is_vegan_friendly BOOLEAN NOT NULL COMMENT '비건 유무 (1. 대상, 0. 비대상)',
    section_id INT NOT NULL COMMENT '매대 정보',
    warehouse_id INT NOT NULL COMMENT '창고 정보',
    FOREIGN KEY (allergy_info_id) REFERENCES allergy_info(allergy_info_id),
    FOREIGN KEY (section_id) REFERENCES section(section_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouse(warehouse_id)
);

-- ================================
-- 9. order
-- ================================
CREATE TABLE `order` (
    order_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '쇼핑 정보 pkey',
    customer_id INT NOT NULL COMMENT '고객 정보',
    start_time DATETIME NOT NULL COMMENT '쇼핑 시작 시간',
    end_time DATETIME NULL COMMENT '쇼핑 종료 시간',
    order_status TINYINT NOT NULL COMMENT '쇼핑 상태 (1. PAID, 2. FAIL_PAID, 3. PICKED_UP, 4. FAIL_PICKUP, 5. CANCELED, 6. RETURNED, 7. FAIL_RETURN, 8. PACKED, 9. FAIL_PACK)',
    failure_reason VARCHAR(50) NULL COMMENT '실패 사유',
    created_at DATETIME NOT NULL COMMENT '주문일자',
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id)
);

-- ================================
-- 10. order_item (수정됨 - order_id FK)
-- ================================
CREATE TABLE order_item (
    order_item_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '쇼핑 상품 정보 pkey',
    order_id INT NOT NULL COMMENT '쇼핑 정보',
    product_id INT NOT NULL COMMENT '상품 정보',
    quantity INT NOT NULL COMMENT '수량',
    created_at DATETIME NOT NULL COMMENT '선택 시간',
    FOREIGN KEY (order_id) REFERENCES `order`(order_id),
    FOREIGN KEY (product_id) REFERENCES product(product_id)
);

-- ================================
-- 11. robot
-- ================================
CREATE TABLE robot (
    robot_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '로봇 정보 pkey',
    robot_type TINYINT NOT NULL COMMENT '로봇 종류 (1. pickee, 2. packee)',
    robot_status TINYINT NOT NULL COMMENT '로봇 상태'
);

-- ================================
-- 12. robot_history
-- ================================
CREATE TABLE robot_history (
    robot_history_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '로봇 작업 이력 pkey',
    robot_id INT NOT NULL COMMENT '로봇 정보',
    history_type TINYINT NOT NULL COMMENT '작업 종류 (1. task, 3. charge, 5. error)',
    order_item_id INT NOT NULL COMMENT '쇼핑 상품 정보',
    is_complete BOOLEAN NOT NULL COMMENT '완료 여부 (1. 완료, 0. 실패)',
    failure_reason VARCHAR(50) NULL COMMENT '실패 사유',
    active_duration INT NOT NULL COMMENT '로봇 사용시간',
    created_at DATETIME NOT NULL COMMENT '등록 일자',
    FOREIGN KEY (robot_id) REFERENCES robot(robot_id),
    FOREIGN KEY (order_item_id) REFERENCES order_item(order_item_id)
);

-- ================================
-- 인덱스 추가 (성능 최적화)
-- ================================
CREATE UNIQUE INDEX idx_customer_id ON customer(id);
CREATE UNIQUE INDEX idx_admin_id ON admin(id);
CREATE UNIQUE INDEX idx_product_barcode ON product(barcode);
CREATE INDEX idx_order_customer_status ON `order`(customer_id, order_status);
CREATE INDEX idx_robot_history_robot_created ON robot_history(robot_id, created_at);