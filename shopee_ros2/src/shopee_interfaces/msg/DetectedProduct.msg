int32 product_id                                    # 상품 ID
float32 confidence                                  # 감지 신뢰도 (0.0~1.0)
shopee_interfaces/BBox bbox                         # Bounding Box (필수)
int32 bbox_number                                   # BBox 번호 (앱 UI 선택용, 미사용 시 0)
shopee_interfaces/DetectionInfo detection_info      # 다각형 영역 정보 (미사용 시 빈 배열)
shopee_interfaces/Pose6D pose                  # 3D 공간 좌표 (그리핑 위치, 미사용 시 0,0,0)
string arm_side