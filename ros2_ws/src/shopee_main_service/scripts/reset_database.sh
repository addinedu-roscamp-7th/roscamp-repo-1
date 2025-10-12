#!/bin/bash
# ========================================
# Shopee Database Reset Script
# ========================================
# 이 스크립트는 기존 데이터를 삭제하고 테이블을 다시 생성합니다.
# (사용자 및 데이터베이스는 그대로 유지)
#
# 실행 방법:
#   chmod +x reset_database.sh
#   ./reset_database.sh

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
    echo "║         Shopee Database Reset Script                     ║"
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

# ========================================
# 메인 실행
# ========================================

print_header

print_warning "이 작업은 모든 데이터를 삭제합니다!"
echo -n "계속하시겠습니까? (y/N): "
read -r confirmation

if [[ ! "$confirmation" =~ ^[Yy]$ ]]; then
    echo "작업이 취소되었습니다."
    exit 0
fi

# 1. 데이터베이스 접속 확인
print_step "데이터베이스 접속 확인..."

if ! mysql -u "${DB_USER}" -p"${DB_PASS}" -e "USE ${DB_NAME};" &>/dev/null; then
    print_error "데이터베이스에 접속할 수 없습니다."
    echo ""
    echo "다음 명령으로 데이터베이스를 먼저 설정하세요:"
    echo "  ./setup_database.sh"
    exit 1
fi
print_success "데이터베이스 접속 성공"

# 2. 테이블 스키마 재생성
print_step "테이블 스키마 재생성 중..."
print_warning "기존 모든 데이터가 삭제됩니다..."

mysql -u "${DB_USER}" -p"${DB_PASS}" "${DB_NAME}" < "$SCRIPT_DIR/init_schema.sql"

if [ $? -eq 0 ]; then
    print_success "테이블 스키마 재생성 완료"
else
    print_error "스키마 생성 실패"
    exit 1
fi

# 3. 샘플 데이터 삽입
print_step "샘플 데이터 삽입 중..."

mysql -u "${DB_USER}" -p"${DB_PASS}" "${DB_NAME}" < "$SCRIPT_DIR/sample_data.sql"

if [ $? -eq 0 ]; then
    print_success "샘플 데이터 삽입 완료"
else
    print_error "데이터 삽입 실패"
    exit 1
fi

# 4. 결과 확인
print_step "데이터베이스 상태 확인..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}데이터베이스 리셋 완료!${NC}"
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
print_success "데이터베이스가 초기 상태로 리셋되었습니다! 🎉"

