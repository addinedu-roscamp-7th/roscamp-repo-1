-- ========================================
-- Shopee Sample Data Insertion
-- ========================================
-- This script inserts sample data for testing

-- ========================================
-- 1. Allergy Info
-- ========================================
INSERT INTO allergy_info (nuts, milk, seafood, soy, peach, gluten, eggs) VALUES
(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),  -- ID: 1 (알러지 없음)
(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),   -- ID: 2 (견과류)
(FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE),   -- ID: 3 (유제품)
(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE);   -- ID: 4 (해산물)

-- ========================================
-- 2. Admin
-- ========================================
INSERT INTO admin (id, password, name) VALUES
('admin', 'admin123', 'PrimaryKey');

-- ========================================
-- 3. Customer
-- ========================================
INSERT INTO customer (id, password, name, gender, age, address, allergy_info_id, is_vegan) VALUES
('user1', 'pass123', '김철수', TRUE, 30, '서울시 강남구', 1, FALSE),
('user2', 'pass123', '이영희', FALSE, 25, '서울시 서초구', 2, FALSE),
('user3', 'pass123', '박해산', TRUE, 35, '서울시 중구', 1, FALSE),
('user4', 'pass123', '최복합', FALSE, 42, '서울시 영등포구', 1, FALSE),
('user5', 'pass123', '정콩글', TRUE, 38, '서울시 동작구', 4, FALSE),
('vegan_user', 'pass123', '김비건', FALSE, 28, '서울시 송파구', 1, TRUE);

-- ========================================
-- 4. Location
-- ========================================
-- Warehouse Locations
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('WAREHOUSE_A', 10.0, 20.0, 0.0, 100),   -- ID: 1 (창고 A)
('WAREHOUSE_B', 12.0, 22.0, 0.0, 110),   -- ID: 2 (창고 B)
('WAREHOUSE_C', 8.0, 18.0, 0.0, 120);    -- ID: 3 (창고 C)

-- Shelf Locations
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('SHELF_A', 15.0, 25.0, 1.57, 101),  -- ID: 4 (Shelf A)
('SHELF_B', 20.0, 30.0, 3.14, 102),  -- ID: 5 (Shelf B)
('SHELF_C', 25.0, 35.0, 0.0, 103);   -- ID: 6 (Shelf C)

-- Section Locations (Shelf A : 기성품)
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('SECTION_A_1', -2.10, 1.50, 0.0, 201),   -- ID: 7 (Section A-1: 과일)
('SECTION_A_2', -2.10, 1.50, 0.0, 202),   -- ID: 8 (Section A-2: 과일)
('SECTION_A_3', -2.10, 1.50, 0.0, 203),   -- ID: 9 (Section A-3: 스프레드)
('SECTION_A_4', -2.10, 1.50, 0.0, 204);   -- ID: 10 (Section A-4: 예비)

-- Section Locations (Shelf B : 신석식품)
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('SECTION_B_1', 0.80, 0.00, 1.57, 211),   -- ID: 11 (Section B-1: 생선)
('SECTION_B_2', 0.80, 0.00, 1.57, 212),   -- ID: 12 (Section B-2: 생선)
('SECTION_B_3', 0.80, 0.00, 1.57, 213),   -- ID: 13 (Section B-3: 생선)
('SECTION_B_4', 0.80, 0.00, 1.57, 214);   -- ID: 14 (Section B-4: 생선)

-- Section Locations (Shelf C : 과자)
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('SECTION_C_1', 3.20, 2.50, -1.57, 221),   -- ID: 15 (Section C-1: 이클립스)
('SECTION_C_2', 3.20, 2.50, -1.57, 222),   -- ID: 16 (Section C-2: 이클립스)
('SECTION_C_3', 3.20, 2.50, -1.57, 223),   -- ID: 17 (Section C-3: 이클립스)
('SECTION_C_4', 3.20, 2.50, -1.57, 224);   -- ID: 18 (Section C-4: 이클립스)

-- Special Locations
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('ROBOT_HOME', 5.0, 5.0, 0.0, 300),     -- ID: 19 (충전 및 대기 장소)
('PACKING_ZONE', 30.0, 40.0, 0.0, 400);   -- ID: 20 (포장 장소)

-- ========================================
-- 5. Warehouse
-- ========================================
INSERT INTO warehouse (location_id, warehouse_name) VALUES
(1, '창고 A'),
(2, '창고 B'),
(3, '창고 C');

-- ========================================
-- 6. Shelf
-- ========================================
INSERT INTO shelf (location_id, shelf_name) VALUES
(4, 'Shelf A'),
(5, 'Shelf B'),
(6, 'Shelf C');

-- ========================================
-- 7. Section
-- ========================================
INSERT INTO section (shelf_id, location_id, section_name) VALUES
-- Shelf A Sections
(1, 7, 'SECTION_A_1'),    -- ID: 1 
(1, 8, 'SECTION_A_2'),    -- ID: 2 
(1, 9, 'SECTION_A_3'),    -- ID: 3 
(1, 10, 'SECTION_A_4'),   -- ID: 4 
-- Shelf B Sections
(2, 11, 'SECTION_B_1'),   -- ID: 5 
(2, 12, 'SECTION_B_2'),   -- ID: 6 
(2, 13, 'SECTION_B_3'),   -- ID: 7 
(2, 14, 'SECTION_B_4'),   -- ID: 8 
-- Shelf C Sections
(3, 15, 'SECTION_C_1'),   -- ID: 9 
(3, 16, 'SECTION_C_2'),   -- ID: 10 
(3, 17, 'SECTION_C_3'),   -- ID: 11 
(3, 18, 'SECTION_C_4');   -- ID: 12

-- ========================================
-- 8. Product (테스트용 상품)
-- ========================================
INSERT INTO product (barcode, name, quantity, price, discount_rate, category, allergy_info_id, is_vegan_friendly, auto_select, section_id, warehouse_id,length,width,height) VALUES
-- Shelf A, Section 1 (기성품)
('7701234567001', '고추냉이', 2, 4500, 0, '기성품', 1, FALSE, TRUE, 1, 1, 2.8, 4.0, 14.1),
('7701234567002', '불닭캔', 3, 3000, 5, '기성품', 1, FALSE, TRUE, 1, 1, 7.7, 7.7, 3.3),
('7701234567003', '버터캔', 3, 3000, 5, '기성품', 1, FALSE, TRUE, 1, 1, 7.7, 7.7, 3.3),
('7701234567004', '리챔', 3, 6000, 5, '기성품', 1, FALSE, TRUE, 1, 1, 6.0, 10.0, 5.5),
('7701234567005', '두유', 3, 1500, 0, '기성품', 1, FALSE, TRUE, 1, 1, 3.9, 5.5, 10.7),
('7701234567006', '카프리썬', 2, 2500, 0, '기성품', 1, FALSE, TRUE, 1, 1, 6.0, 8.0, 14.5),
-- Shelf B, Section 5 (과일)
('8801234567001', '홍사과', 3, 1700, 0, '과일', 1, TRUE, FALSE, 5, 2, 4.0, 4.0, 4.5),
('8801234567002', '청사과', 3, 1700, 0, '과일', 1, TRUE, FALSE, 5, 2, 4.0, 4.0, 4.5),
('8801234567003', '오렌지', 3, 2000, 0, '과일', 1, TRUE, FALSE, 5, 2, 3.5, 3.5, 2.5),
-- Shelf B, Section 7 (육류/어류)
('8801234567004', '삼겹살', 3, 8000, 0, '육류', 1, FALSE, FALSE, 7, 2, 8.0, 16.0, 3.0),
('8801234567005', '닭', 3, 7500, 0, '육류', 1, FALSE, FALSE, 7, 2, 7.0, 13.0, 3.0),
('8801234567006', '생선', 3, 8000, 0, '어류', 4, FALSE, FALSE, 7, 2, 5.0, 18.0, 3.0),
('8801234567007', '전복', 3, 15000, 0, '어류', 4, FALSE, FALSE, 7, 2, 5.5, 9.0, 3.0),
-- Shelf C, Section 11 (과자/베이커리)
('9901234567001', '이클립스', 3, 1000, 0, '과자', 3, FALSE, TRUE, 11, 3, 3.9, 4.0, 7.8),
('9901234567002', '아이비', 3, 3000, 5, '과자', 3, FALSE, TRUE, 11, 3, 5.0, 10.0, 5.0),
('9901234567003', '빼빼로', 3, 1500, 0, '과자', 2, FALSE, TRUE, 11, 3, 2.3, 8.7, 16.0),
('9901234567004', '오예스', 3, 3500, 0, '과자', 3, FALSE, TRUE, 11, 3, 7.5, 9.5, 2.5);

-- ========================================
-- 9. Box (포장 박스 규격)
-- ========================================
INSERT INTO box (width, depth, height) VALUES
(15, 10, 5),   -- Small
(25, 20, 15),  -- Medium
(40, 30, 25),  -- Large
(50, 40, 30);  -- Extra Large

-- ========================================
-- 10. Robot (피킹/포장 로봇)
-- ========================================
INSERT INTO robot (robot_type) VALUES
(1),  -- ID: 1, Type: Pickee(1)
(1),  -- ID: 2, Type: Pickee(1)
(2),  -- ID: 3, Type: Packee(2)
(2);  -- ID: 4, Type: Packee(2)

-- ========================================
-- Sample Data Insertion Complete
-- ========================================
--
-- Summary:
-- - 4 Allergy Info entries
--   * ID 1: 알러지없음
--   * ID 2: 견과류
--   * ID 3: 유제품
--   * ID 4: 해산물
--
-- - 1 Admin user (admin/admin123) - 대시보드 관리용
-- - 6 Customers - 쇼핑 고객용
--   * user1 (알러지없음), user2 (견과류), user3 (해산물)
--   * user4 (알러지없음), user5 (해산물)
--   * vegan_user (알러지없음, 비건)
--
-- - 20 Locations
--   * 창고: 3개 (창고 A, B, C)
--   * 선반: 3개 (Shelf A, B, C)
--   * 섹션: 12개 (각 선반마다 4개씩)
--   * 특수: 2개 (충전/대기 장소, 포장 장소)
--
-- - 3 Warehouses (창고 A, B, C)
-- - 3 Shelves (Shelf A, B, C)
-- - 12 Sections
--   * Shelf A: SECTION_A_1~4 (기성품, 예비, 예비, 예비)
--   * Shelf B: SECTION_B_1~4 (과일, 예비, 육류/어류, 예비)
--   * Shelf C: SECTION_C_1~4 (예비, 예비, 과자/베이커리, 예비)
--
-- - 17 Products
--   * Shelf A (Section 1, Warehouse 1): 기성품 6개 (고추냉이, 불닭캔, 버터캔, 리챔, 두유, 카프리썬)
--   * Shelf B (Section 5, Warehouse 2): 과일 3개 (홍사과, 청사과, 오렌지)
--   * Shelf B (Section 7, Warehouse 2): 육류/어류 4개 (삼겹살, 닭, 생선, 전복)
--   * Shelf C (Section 11, Warehouse 3): 과자/베이커리 4개 (이클립스, 아이비, 빼빼로, 오예스)
--
-- - 4 Boxes (S, M, L, XL)
-- - 4 Robots (Pickee 2대, Packee 2대)
