# Main ↔ DB Manager

**Main** = Shopee Main Service

**DB Manager** = Database Management System

## 개요

Main Service와 DB Manager 간의 인터페이스는 직접적인 데이터베이스 쿼리 또는 ORM(Object-Relational Mapping)을 통해 이루어진다. 이 문서에서는 네트워크 프로토콜이 아닌, 내부 함수 호출 관점에서의 인터페이스를 기술한다.

## 함수 인터페이스

### IF_DB_01: 상품 위치 조회

**Description:** 상품 ID를 기반으로 데이터베이스에서 상품의 위치(창고, 매대) 정보를 조회한다.

| 항목 | 내용 |
|---|---|
| **호출 주체** | Main Service |
| **피호출 객체** | DB Manager |
| **함수명** | `getProductLocation` |

#### 파라미터
- `product_id` (string): 위치를 조회할 상품의 고유 ID

#### 반환값
- `ProductLocation` (object) 또는 `null` (상품 정보가 없을 경우)

**ProductLocation 객체 구조:**
- `product_id` (string)
- `location_id` (string): 창고 또는 구역 ID
- `shelf_id` (string): 매대 ID
- `stock_quantity` (int): 현재 재고 수량

#### 예시
**호출:**

```javascript
const location = getProductLocation("PROD_005");
```

**반환값 (성공):**

```json
{
  "product_id": "PROD_005",
  "location_id": "LOC_B2",
  "shelf_id": "SHELF_B2_03",
  "stock_quantity": 15
}
```

**반환값 (실패/정보 없음):**

```javascript
null
```
