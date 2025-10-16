# Shopee Database Scripts

데이터베이스 초기화 및 관리를 위한 스크립트 모음입니다.

## 📁 파일 설명

### SQL 스크립트
- **`init_schema.sql`** - 테이블 스키마 정의 (DROP + CREATE)
- **`sample_data.sql`** - 테스트용 샘플 데이터 (INSERT)

### Bash 스크립트
- **`setup_database.sh`** - 🔧 **최초 설정용** (사용자/DB 생성 + 스키마 + 데이터)
- **`reset_database.sh`** - 🔄 **리셋용** (데이터 삭제 후 재생성)

### 테스트 스크립트
- **`test_client.py`** - Main Service TCP API 테스트 클라이언트

## 🚀 사용 방법

### 1. 최초 설정 (처음 한 번만)

데이터베이스를 처음 설정할 때 사용합니다.

```bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service/scripts
./setup_database.sh
```

**수행 작업:**
- ✅ MySQL 사용자 `shopee` 생성 (비밀번호: `shopee`)
- ✅ 데이터베이스 `shopee` 생성
- ✅ 권한 부여
- ✅ 테이블 스키마 생성 (12개 테이블)
- ✅ 샘플 데이터 삽입 (고객, 상품, 로봇 등)

**주의:** sudo 권한이 필요할 수 있습니다.

```bash
# sudo가 필요한 경우
sudo ./setup_database.sh
```

### 2. 데이터베이스 리셋 (개발 중)

테스트로 데이터가 망가졌을 때, 초기 상태로 되돌립니다.

```bash
./reset_database.sh
```

**수행 작업:**
- ⚠️ 모든 데이터 삭제 (확인 메시지 표시)
- ✅ 테이블 재생성
- ✅ 샘플 데이터 재삽입

**사용자/DB는 그대로 유지됩니다.**

### 3. 수동 실행 (고급 사용자)

```bash
# 1. 스키마만 재생성
mysql -u shopee -pshopee shopee < init_schema.sql

# 2. 샘플 데이터만 삽입
mysql -u shopee -pshopee shopee < sample_data.sql

# 3. 특정 쿼리 실행
mysql -u shopee -pshopee shopee -e "SELECT * FROM product;"
```

## 📊 생성되는 데이터

### 테이블 (12개)
- allergy_info
- admin
- customer
- location
- warehouse
- shelf
- section
- product
- `order`
- order_item
- robot
- robot_history

### 샘플 데이터
| 항목 | 개수 | 설명 |
|------|------|------|
| 관리자 | 1 | admin/admin123 |
| 고객 | 4 | admin, user1, user2, vegan_user |
| 상품 | 8 | 과일 3, 채소 2, 음료 3 |
| 로봇 | 3 | Pickee 2대, Packee 1대 |
| 위치 | 10 | 창고, 선반, 섹션 위치 |

## 🔌 Main Service 연동

Main Service가 데이터베이스를 사용하려면 `.env` 파일을 설정하세요:

```bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service

# .env 파일 생성
cat > .env << 'EOF'
SHOPEE_DB_URL=mysql+pymysql://shopee:shopee@localhost:3306/shopee
SHOPEE_LLM_BASE_URL=http://localhost:8000
SHOPEE_API_HOST=0.0.0.0
SHOPEE_API_PORT=5000
SHOPEE_LOG_LEVEL=INFO
EOF
```

## 🧪 테스트 워크플로우

```bash
# 1. 데이터베이스 설정
./setup_database.sh

# 2. Mock 환경 실행 (3개 터미널)
# 터미널 1 - Mock LLM Server
ros2 run shopee_main_service mock_llm_server

# 터미널 2 - Mock Robot Node (Pickee/Packee 또는 선택적 실행)
ros2 run shopee_main_service mock_robot_node
# ros2 run shopee_main_service mock_robot_node --mode pickee
# ros2 run shopee_main_service mock_robot_node --mode packee

# 터미널 3 - Main Service
ros2 run shopee_main_service main_service_node

# 3. 테스트 실행 (새 터미널)
cd scripts
python3 test_client.py              # 자동 모드
python3 test_client.py -i           # 인터랙티브 모드
python3 test_client.py inventory    # 재고 관리 테스트
```

## 🔍 데이터베이스 접속

```bash
# MySQL CLI 접속
mysql -u shopee -pshopee shopee

# 테이블 확인
mysql -u shopee -pshopee shopee -e "SHOW TABLES;"

# 상품 조회
mysql -u shopee -pshopee shopee -e "SELECT * FROM product;"

# 주문 확인
mysql -u shopee -pshopee shopee -e "SELECT * FROM \`order\`;"
```

## 📝 트러블슈팅

### MySQL 접속 실패
```bash
ERROR 1698 (28000): Access denied for user 'root'@'localhost'
```
→ `sudo ./setup_database.sh` 실행

### 스키마 생성 실패
```bash
ERROR 1005 (HY000): Can't create table (errno: 150)
```
→ Foreign Key 순서 문제. `reset_database.sh` 실행

### 테이블이 이미 존재
```bash
ERROR 1050 (42S01): Table 'product' already exists
```
→ `DROP TABLE IF EXISTS` 때문에 정상. 무시해도 됨.

## 📂 최종 파일 구조

```
scripts/
├── setup_database.sh      # DB 초기 설정
├── reset_database.sh      # DB 리셋
├── init_schema.sql        # 테이블 스키마
├── sample_data.sql        # 샘플 데이터
├── test_client.py         # API 테스트 클라이언트
└── README.md              # 이 문서
```

## 🔗 관련 문서

- [TEST_GUIDE.md](../TEST_GUIDE.md) - 전체 테스트 가이드
- [README.md](../README.md) - Main Service 설명
- [ERDiagram.md](../../../../docs/ERDiagram/ERDiagram.md) - 데이터베이스 설계

---

**마지막 업데이트:** 2025-10-12
