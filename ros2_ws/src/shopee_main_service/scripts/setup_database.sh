#!/bin/bash
# ========================================
# Shopee Database Setup Script
# ========================================
# 이 스크립트는 Shopee 데이터베이스를 처음부터 설정합니다.
#
# 실행 방법:
#   chmod +x setup_database.sh
#   ./setup_database.sh

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 설정
DB_USER="shopee"
DB_PASS="shopee"
DB_NAME="shopee"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================
# 함수 정의
# ========================================

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║         Shopee Database Setup Script                     ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# MySQL 접속 테스트
check_mysql_root() {
    print_step "MySQL root 접속 확인 중..."
    
    if sudo mysql -u root -e "SELECT 1;" &>/dev/null; then
        print_success "MySQL root 접속 가능 (sudo 사용)"
        return 0
    elif mysql -u root -e "SELECT 1;" &>/dev/null; then
        print_success "MySQL root 접속 가능"
        return 0
    else
        print_error "MySQL root 접속 실패"
        echo ""
        echo "다음 중 하나를 시도하세요:"
        echo "  1. sudo ./setup_database.sh"
        echo "  2. MySQL root 비밀번호 설정 확인"
        return 1
    fi
}

# MySQL 명령 실행 (root)
mysql_root() {
    if sudo mysql -u root -e "SELECT 1;" &>/dev/null; then
        sudo mysql -u root "$@"
    else
        mysql -u root "$@"
    fi
}

# ========================================
# 메인 실행
# ========================================

print_header

# 1. MySQL 확인
print_step "MySQL 설치 확인..."
if ! command -v mysql &> /dev/null; then
    print_error "MySQL이 설치되어 있지 않습니다."
    echo "설치 명령: sudo apt install mysql-server"
    exit 1
fi
print_success "MySQL 설치됨"

check_mysql_root || exit 1

# 2. 사용자 및 데이터베이스 생성
print_step "데이터베이스 사용자 및 DB 생성 중..."

mysql_root << EOF
-- 기존 사용자 및 데이터베이스 확인 후 생성
CREATE USER IF NOT EXISTS '${DB_USER}'@'localhost' IDENTIFIED BY '${DB_PASS}';
CREATE DATABASE IF NOT EXISTS ${DB_NAME};
GRANT ALL PRIVILEGES ON ${DB_NAME}.* TO '${DB_USER}'@'localhost';
FLUSH PRIVILEGES;
EOF

if [ $? -eq 0 ]; then
    print_success "사용자 '${DB_USER}' 및 데이터베이스 '${DB_NAME}' 생성 완료"
else
    print_error "사용자/DB 생성 실패"
    exit 1
fi

# 3. 테이블 스키마 생성
print_step "테이블 스키마 생성 중..."
print_warning "기존 테이블이 있다면 삭제됩니다."

if [ ! -f "$SCRIPT_DIR/init_schema.sql" ]; then
    print_error "init_schema.sql 파일을 찾을 수 없습니다: $SCRIPT_DIR/init_schema.sql"
    exit 1
fi

mysql -u "${DB_USER}" -p"${DB_PASS}" "${DB_NAME}" < "$SCRIPT_DIR/init_schema.sql"

if [ $? -eq 0 ]; then
    print_success "테이블 스키마 생성 완료"
else
    print_error "스키마 생성 실패"
    exit 1
fi

# 4. 샘플 데이터 삽입
print_step "샘플 데이터 삽입 중..."

if [ ! -f "$SCRIPT_DIR/sample_data.sql" ]; then
    print_error "sample_data.sql 파일을 찾을 수 없습니다: $SCRIPT_DIR/sample_data.sql"
    exit 1
fi

mysql -u "${DB_USER}" -p"${DB_PASS}" "${DB_NAME}" < "$SCRIPT_DIR/sample_data.sql"

if [ $? -eq 0 ]; then
    print_success "샘플 데이터 삽입 완료"
else
    print_error "데이터 삽입 실패"
    exit 1
fi

# 5. 결과 확인
print_step "데이터베이스 상태 확인..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}데이터베이스 설정 완료!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mysql -u "${DB_USER}" -p"${DB_PASS}" "${DB_NAME}" << 'EOF'
SELECT 
    '테이블' as '항목',
    COUNT(*) as '개수'
FROM information_schema.tables 
WHERE table_schema = 'shopee'
UNION ALL
SELECT '고객 (Customer)', COUNT(*) FROM customer
UNION ALL
SELECT '관리자 (Admin)', COUNT(*) FROM admin
UNION ALL
SELECT '상품 (Product)', COUNT(*) FROM product
UNION ALL
SELECT '로봇 (Robot)', COUNT(*) FROM robot
UNION ALL
SELECT '위치 (Location)', COUNT(*) FROM location
UNION ALL
SELECT '창고 (Warehouse)', COUNT(*) FROM warehouse
UNION ALL
SELECT '선반 (Shelf)', COUNT(*) FROM shelf
UNION ALL
SELECT '섹션 (Section)', COUNT(*) FROM section;
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}접속 정보:${NC}"
echo "  사용자명: ${DB_USER}"
echo "  비밀번호: ${DB_PASS}"
echo "  데이터베이스: ${DB_NAME}"
echo ""
echo -e "${BLUE}접속 명령:${NC}"
echo "  mysql -u ${DB_USER} -p${DB_PASS} ${DB_NAME}"
echo ""
echo -e "${BLUE}.env 파일 설정:${NC}"
echo "  SHOPEE_DB_URL=mysql+pymysql://${DB_USER}:${DB_PASS}@localhost:3306/${DB_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_success "모든 설정이 완료되었습니다! 🎉"

