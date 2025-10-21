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
(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE),   -- ID: 4 (해산물)
(FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE),   -- ID: 5 (콩)
(FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE),   -- ID: 6 (복숭아)
(FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE),   -- ID: 7 (글루텐)
(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE),   -- ID: 8 (계란)
(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE),    -- ID: 9 (견과류 + 유제품)
(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE),    -- ID: 10 (견과류 + 해산물)
(FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE),    -- ID: 11 (유제품 + 해산물)
(FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE),    -- ID: 12 (콩 + 글루텐)
(FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE),    -- ID: 13 (유제품 + 계란)
(TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE);     -- ID: 14 (견과류 + 유제품 + 콩)

-- ========================================
-- 2. Admin
-- ========================================
INSERT INTO admin (id, password, name) VALUES
('admin', 'admin123', '장진혁');

-- ========================================
-- 3. Customer
-- ========================================
INSERT INTO customer (id, password, name, gender, age, address, allergy_info_id, is_vegan) VALUES
('admin', 'admin123', '장진혁', TRUE, 28, '서울시 강서구', 1, FALSE),
('user1', 'pass123', '김철수', TRUE, 30, '서울시 강남구', 1, FALSE),
('user2', 'pass123', '이영희', FALSE, 25, '서울시 서초구', 2, FALSE),
('user3', 'pass123', '박해산', TRUE, 35, '서울시 중구', 4, FALSE),
('user4', 'pass123', '최복합', FALSE, 42, '서울시 영등포구', 9, FALSE),
('user5', 'pass123', '정콩글', TRUE, 38, '서울시 동작구', 12, FALSE),
('vegan_user', 'pass123', '박비건', FALSE, 28, '서울시 송파구', 1, TRUE);

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

-- Section Locations (Shelf A)
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('SECTION_A_1', 15.0, 25.5, 0.0, 201),   -- ID: 7 (Section A-1: 과일)
('SECTION_A_2', 15.5, 25.5, 0.0, 202),   -- ID: 8 (Section A-2: 과일)
('SECTION_A_3', 16.0, 25.5, 0.0, 203),   -- ID: 9 (Section A-3: 스프레드)
('SECTION_A_4', 16.5, 25.5, 0.0, 204);   -- ID: 10 (Section A-4: 예비)

-- Section Locations (Shelf B)
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('SECTION_B_1', 20.0, 30.5, 0.0, 211),   -- ID: 11 (Section B-1: 채소)
('SECTION_B_2', 20.5, 30.5, 0.0, 212),   -- ID: 12 (Section B-2: 채소)
('SECTION_B_3', 21.0, 30.5, 0.0, 213),   -- ID: 13 (Section B-3: 신선식품)
('SECTION_B_4', 21.5, 30.5, 0.0, 214);   -- ID: 14 (Section B-4: 예비)

-- Section Locations (Shelf C)
INSERT INTO location (location_name, location_x, location_y, location_theta, aruco_marker) VALUES
('SECTION_C_1', 25.0, 35.5, 0.0, 221),   -- ID: 15 (Section C-1: 음료)
('SECTION_C_2', 25.5, 35.5, 0.0, 222),   -- ID: 16 (Section C-2: 음료)
('SECTION_C_3', 26.0, 35.5, 0.0, 223),   -- ID: 17 (Section C-3: 베이커리)
('SECTION_C_4', 26.5, 35.5, 0.0, 224);   -- ID: 18 (Section C-4: 예비)

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
(1, 7, 'SECTION_A_1'),    -- ID: 1 (과일)
(1, 8, 'SECTION_A_2'),    -- ID: 2 (과일)
(1, 9, 'SECTION_A_3'),    -- ID: 3 (스프레드)
(1, 10, 'SECTION_A_4'),   -- ID: 4 (예비)
-- Shelf B Sections
(2, 11, 'SECTION_B_1'),   -- ID: 5 (채소)
(2, 12, 'SECTION_B_2'),   -- ID: 6 (채소)
(2, 13, 'SECTION_B_3'),   -- ID: 7 (신선식품)
(2, 14, 'SECTION_B_4'),   -- ID: 8 (예비)
-- Shelf C Sections
(3, 15, 'SECTION_C_1'),   -- ID: 9 (음료)
(3, 16, 'SECTION_C_2'),   -- ID: 10 (음료)
(3, 17, 'SECTION_C_3'),   -- ID: 11 (베이커리)
(3, 18, 'SECTION_C_4');   -- ID: 12 (예비)

-- ========================================
-- 8. Product (테스트용 상품)
-- ========================================
INSERT INTO product (barcode, name, quantity, price, discount_rate, category, allergy_info_id, is_vegan_friendly, auto_select, section_id, warehouse_id) VALUES
-- Shelf A (과일/스프레드)
('8801234567001', '사과', 50, 3000, 0, '과일', 1, TRUE, TRUE, 1, 3),
('8801234567002', '바나나', 30, 2000, 10, '과일', 1, TRUE, TRUE, 2, 3),
('8801234567003', '유기농 사과', 20, 5000, 0, '과일', 1, TRUE, TRUE, 1, 3),
('8801234567010', '땅콩버터', 25, 4500, 0, '스프레드', 2, TRUE, FALSE, 3, 1),

-- Shelf B (채소/신선식품)
('8801234567004', '양상추', 40, 2500, 0, '채소', 1, TRUE, TRUE, 5, 3),
('8801234567005', '토마토', 35, 3500, 5, '채소', 1, TRUE, TRUE, 6, 3),
('8801234567011', '계란', 100, 5000, 0, '신선식품', 8, FALSE, FALSE, 7, 3),
('8801234567012', '새우', 20, 12000, 15, '해산물', 4, FALSE, FALSE, 7, 2),

-- Shelf C (음료/베이커리)
('8801234567006', '사과주스', 60, 1500, 0, '음료', 1, TRUE, TRUE, 9, 1),
('8801234567007', '우유', 45, 2500, 0, '음료', 3, FALSE, TRUE, 10, 2),
('8801234567008', '두유', 40, 2300, 0, '음료', 5, TRUE, TRUE, 10, 2),
('8801234567009', '아몬드우유', 35, 2800, 0, '음료', 9, TRUE, TRUE, 10, 2),
('8801234567013', '식빵', 30, 3500, 0, '베이커리', 7, FALSE, FALSE, 11, 1);

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
-- - 14 Allergy Info entries (단일 8개, 복합 6개)
--   * 단일: 알러지없음, 견과류, 유제품, 해산물, 콩, 복숭아, 글루텐, 계란
--   * 복합: 견과류+유제품, 견과류+해산물, 유제품+해산물, 콩+글루텐, 유제품+계란, 견과류+유제품+콩
--
-- - 1 Admin user (admin/admin123)
-- - 7 Customers
--   * admin (알러지없음), user1 (알러지없음), user2 (견과류)
--   * user3 (해산물), user4 (견과류+유제품), user5 (콩+글루텐)
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
--   * Shelf A: SECTION_A_1~4 (과일, 과일, 스프레드, 예비)
--   * Shelf B: SECTION_B_1~4 (채소, 채소, 신선식품, 예비)
--   * Shelf C: SECTION_C_1~4 (음료, 음료, 베이커리, 예비)
--
-- - 13 Products (과일 3개, 채소 2개, 음료 4개, 기타 4개)
--   * 알러지: 견과류 2개, 유제품 2개, 콩 1개, 해산물 1개, 계란 1개, 글루텐 1개, 견과류+유제품 1개
--   * 창고 A: 사과주스, 땅콩버터, 식빵
--   * 창고 B: 우유, 두유, 아몬드우유, 새우
--   * 창고 C: 사과, 바나나, 유기농 사과, 양상추, 토마토, 계란
--
-- - 4 Boxes (S, M, L, XL)
-- - 4 Robots (Pickee 2대, Packee 2대)
