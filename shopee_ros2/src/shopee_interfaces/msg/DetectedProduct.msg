# 상품 감지 결과 (Pickee, Packee 공통 사용)
# 각 시스템에서 사용하지 않는 필드는 기본값(0, 빈 배열 등)으로 설정

# 필수 필드 (공통)
int32 product_id                                    # 상품 ID
float32 confidence                                  # 감지 신뢰도 (0.0~1.0)
shopee_interfaces/BBox bbox                         # Bounding Box (필수)

# 선택적 필드 - Pickee 전용 (매대 상품 인식)
int32 bbox_number                                   # BBox 번호 (앱 UI 선택용, 미사용 시 0)
shopee_interfaces/DetectionInfo detection_info      # 다각형 영역 정보 (미사용 시 빈 배열)

# 선택적 필드 - Packee 전용 (장바구니 상품 포장)
shopee_interfaces/Point3D position                  # 3D 공간 좌표 (그리핑 위치, 미사용 시 0,0,0)

# 사용 규칙:
# Pickee: product_id, confidence, bbox, bbox_number, detection_info 사용
#         position은 (0, 0, 0)으로 설정
# Packee: product_id, confidence, bbox, position 사용
#         bbox_number는 0, detection_info는 빈 배열로 설정
