-- ========================================
-- Shopee Sample Data Insertion
-- ========================================
-- This script inserts sample data for testing

-- ========================================
-- 1. Allergy Info
-- ========================================
INSERT INTO allergy_info (nuts, milk, seafood, soy, peach, gluten, eggs) VALUES
(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),  -- ID: 1 (알러지 없음)
(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),   -- ID: 2 (견과류 알러지)
(FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE);   -- ID: 3 (유제품 알러지)

-- ========================================
-- 2. Admin
-- ========================================
INSERT INTO admin (id, password, name) VALUES
('admin', 'admin123', '관리자');

-- ========================================
-- 3. Customer
-- ========================================
INSERT INTO customer (id, password, name, gender, age, address, allergy_info_id, is_vegan) VALUES
('admin', 'admin123', '관리자', TRUE, 35, '서울시 중구', 1, FALSE),
('user1', 'pass123', '김철수', TRUE, 30, '서울시 강남구', 1, FALSE),
('user2', 'pass123', '이영희', FALSE, 25, '서울시 서초구', 2, TRUE),
('vegan_user', 'pass123', '박비건', FALSE, 28, '서울시 송파구', 1, TRUE);

-- ========================================
-- 4. Location (창고/선반 위치)
-- ========================================
INSERT INTO location (location_x, location_y, location_theta, aruco_marker) VALUES
(10.0, 20.0, 0.0, 100),  -- ID: 1 (창고 위치)
(15.0, 25.0, 1.57, 101),  -- ID: 2 (선반1 위치)
(20.0, 30.0, 3.14, 102),  -- ID: 3 (선반2 위치)
(25.0, 35.0, 0.0, 103);  -- ID: 4 (선반3 위치)

-- Section Locations
INSERT INTO location (location_x, location_y, location_theta, aruco_marker) VALUES
(15.5, 25.0, 0.0, 201), -- ID: 5 (사과 구역)
(16.0, 25.0, 0.0, 202), -- ID: 6 (바나나 구역)
(20.5, 30.0, 0.0, 203), -- ID: 7 (양상추 구역)
(21.0, 30.0, 0.0, 204), -- ID: 8 (토마토 구역)
(25.5, 35.0, 0.0, 205), -- ID: 9 (주스 구역)
(26.0, 35.0, 0.0, 206); -- ID: 10 (우유 구역)

-- ========================================
-- 5. Warehouse
-- ========================================
INSERT INTO warehouse (location_id, warehouse_name) VALUES
(1, '메인 창고');

-- ========================================
-- 6. Shelf
-- ========================================
INSERT INTO shelf (location_id, shelf_name) VALUES
(2, '과일 선반'),
(3, '채소 선반'),
(4, '음료 선반');

-- ========================================
-- 7. Section
-- ========================================
INSERT INTO section (shelf_id, location_id, section_name) VALUES
(1, 5, '사과 구역'),     -- ID: 1
(1, 6, '바나나 구역'),   -- ID: 2
(2, 7, '양상추 구역'),   -- ID: 3
(2, 8, '토마토 구역'),   -- ID: 4
(3, 9, '주스 구역'),     -- ID: 5
(3, 10, '우유 구역');    -- ID: 6

-- ========================================
-- 8. Product (테스트용 상품)
-- ========================================
INSERT INTO product (barcode, name, quantity, price, discount_rate, category, allergy_info_id, is_vegan_friendly, section_id, warehouse_id) VALUES
-- 과일
('8801234567001', '사과', 50, 3000, 0, '과일', 1, TRUE, 1, 1),
('8801234567002', '바나나', 30, 2000, 10, '과일', 1, TRUE, 2, 1),
('8801234567003', '유기농 사과', 20, 5000, 0, '과일', 1, TRUE, 1, 1),

-- 채소
('8801234567004', '양상추', 40, 2500, 0, '채소', 1, TRUE, 3, 1),
('8801234567005', '토마토', 35, 3500, 5, '채소', 1, TRUE, 4, 1),

-- 음료
('8801234567006', '사과주스', 60, 1500, 0, '음료', 1, TRUE, 5, 1),
('8801234567007', '우유', 45, 2500, 0, '음료', 3, FALSE, 6, 1),
('8801234567008', '두유', 40, 2300, 0, '음료', 1, TRUE, 6, 1);

-- ========================================
-- 9. Robot (피킹/포장 로봇)
-- ========================================
INSERT INTO robot (robot_type, robot_status) VALUES
(1, 0),  -- ID: 1, Type: Pickee(1), Status: IDLE(0)
(1, 0),  -- ID: 2, Type: Pickee(1), Status: IDLE(0)
(2, 0);  -- ID: 3, Type: Packee(2), Status: IDLE(0)
(2, 0);  -- ID: 4, Type: Packee(2), Status: IDLE(0)

-- ========================================
-- Sample Data Insertion Complete
-- ========================================
--
-- Summary:
-- - 3 Allergy Info entries
-- - 1 Admin user (admin/admin123)
-- - 3 Customers (user1/pass123, user2/pass123, vegan_user/pass123)
-- - 4 Locations
-- - 1 Warehouse
-- - 3 Shelves
-- - 6 Sections
-- - 8 Products (과일 3개, 채소 2개, 음료 3개)
-- - 3 Robots (Pickee 2대, Packee 1대)
