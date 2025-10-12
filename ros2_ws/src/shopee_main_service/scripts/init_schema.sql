-- ========================================
-- Shopee Database Schema Initialization
-- ========================================
-- This script creates all tables based on database_models.py

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS robot_history;
DROP TABLE IF EXISTS order_item;
DROP TABLE IF EXISTS `order`;
DROP TABLE IF EXISTS robot;
DROP TABLE IF EXISTS product;
DROP TABLE IF EXISTS section;
DROP TABLE IF EXISTS shelf;
DROP TABLE IF EXISTS warehouse;
DROP TABLE IF EXISTS location;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS admin;
DROP TABLE IF EXISTS allergy_info;

-- ========================================
-- 1. Allergy Info Table
-- ========================================
CREATE TABLE allergy_info (
    allergy_info_id INT AUTO_INCREMENT PRIMARY KEY,
    nuts BOOLEAN NOT NULL,
    milk BOOLEAN NOT NULL,
    seafood BOOLEAN NOT NULL,
    soy BOOLEAN NOT NULL,
    peach BOOLEAN NOT NULL,
    gluten BOOLEAN NOT NULL,
    eggs BOOLEAN NOT NULL
);

-- ========================================
-- 2. Admin Table
-- ========================================
CREATE TABLE admin (
    admin_id INT AUTO_INCREMENT PRIMARY KEY,
    id VARCHAR(20) NOT NULL UNIQUE,
    password VARCHAR(256) NOT NULL,
    name VARCHAR(10) NOT NULL
);

-- ========================================
-- 3. Customer Table
-- ========================================
CREATE TABLE customer (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    id VARCHAR(20) NOT NULL UNIQUE,
    password VARCHAR(256) NOT NULL,
    name VARCHAR(10) NOT NULL,
    gender BOOLEAN NOT NULL,
    age INT NOT NULL,
    address VARCHAR(256) NOT NULL,
    allergy_info_id INT NOT NULL,
    is_vegan BOOLEAN NOT NULL,
    FOREIGN KEY (allergy_info_id) REFERENCES allergy_info(allergy_info_id)
);

-- ========================================
-- 4. Location Table
-- ========================================
CREATE TABLE location (
    location_id INT AUTO_INCREMENT PRIMARY KEY,
    location_x FLOAT NOT NULL,
    location_y FLOAT NOT NULL,
    aruco_marker INT NOT NULL
);

-- ========================================
-- 5. Warehouse Table
-- ========================================
CREATE TABLE warehouse (
    warehouse_id INT AUTO_INCREMENT PRIMARY KEY,
    location_id INT NOT NULL,
    warehouse_name VARCHAR(50) NOT NULL,
    FOREIGN KEY (location_id) REFERENCES location(location_id)
);

-- ========================================
-- 6. Shelf Table
-- ========================================
CREATE TABLE shelf (
    shelf_id INT AUTO_INCREMENT PRIMARY KEY,
    location_id INT NOT NULL,
    shelf_name VARCHAR(50) NOT NULL,
    FOREIGN KEY (location_id) REFERENCES location(location_id)
);

-- ========================================
-- 7. Section Table
-- ========================================
CREATE TABLE section (
    section_id INT AUTO_INCREMENT PRIMARY KEY,
    shelf_id INT NOT NULL,
    location_id INT NOT NULL,
    section_name VARCHAR(50) NOT NULL,
    FOREIGN KEY (shelf_id) REFERENCES shelf(shelf_id),
    FOREIGN KEY (location_id) REFERENCES location(location_id)
);

-- ========================================
-- 8. Product Table
-- ========================================
CREATE TABLE product (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    barcode VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(50) NOT NULL,
    quantity INT NOT NULL,
    price INT NOT NULL,
    discount_rate INT NOT NULL,
    category VARCHAR(10) NOT NULL,
    allergy_info_id INT NOT NULL,
    is_vegan_friendly BOOLEAN NOT NULL,
    section_id INT NOT NULL,
    warehouse_id INT NOT NULL,
    FOREIGN KEY (allergy_info_id) REFERENCES allergy_info(allergy_info_id),
    FOREIGN KEY (section_id) REFERENCES section(section_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouse(warehouse_id)
);

-- ========================================
-- 9. Order Table
-- ========================================
CREATE TABLE `order` (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    order_status TINYINT NOT NULL,
    failure_reason VARCHAR(50),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id)
);

-- ========================================
-- 10. Order Item Table
-- ========================================
CREATE TABLE order_item (
    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES `order`(order_id),
    FOREIGN KEY (product_id) REFERENCES product(product_id)
);

-- ========================================
-- 11. Robot Table
-- ========================================
CREATE TABLE robot (
    robot_id INT AUTO_INCREMENT PRIMARY KEY,
    robot_type TINYINT NOT NULL,
    robot_status TINYINT NOT NULL
);

-- ========================================
-- 12. Robot History Table
-- ========================================
CREATE TABLE robot_history (
    robot_history_id INT AUTO_INCREMENT PRIMARY KEY,
    robot_id INT NOT NULL,
    history_type TINYINT NOT NULL,
    order_item_id INT NOT NULL,
    is_complete BOOLEAN NOT NULL,
    failure_reason VARCHAR(50),
    active_duration INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (robot_id) REFERENCES robot(robot_id),
    FOREIGN KEY (order_item_id) REFERENCES order_item(order_item_id)
);

-- ========================================
-- Schema Creation Complete
-- ========================================
