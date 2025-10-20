# shopee_interfaces 변경 제안 (최종안 v4)

**사유:** YOLOv8-seg 모델의 결과(Polygon, BBox)를 모두 전달하고, 메시지의 의미를 명확히 하기 위해 인터페이스를 확장합니다.

---

## 필요한 조치

1.  **신규 메시지 파일 1개 생성** (`DetectionInfo.msg`)
2.  **기존 메시지 파일 1개 수정** (`PickeeDetectedProduct.msg`)

---

## 1. 신규 메시지: `DetectionInfo.msg`

객체 하나의 기하학적 정보(외곽선+BBox)를 모두 담는 새로운 메시지를 만듭니다.

- **생성할 파일:** `shopee_interfaces/msg/DetectionInfo.msg`
- **파일 내용:**
  ```
  # 하나의 감지된 객체에 대한 기하학적 정보
  shopee_interfaces/Point2D[] polygon      # 정밀한 외곽선 정보
  shopee_interfaces/BBox bbox_coords  # 간편한 사각형 정보
  ```

## 2. 기존 메시지 수정: `PickeeDetectedProduct.msg`

감지된 상품 정보가 `bbox_number`는 유지하면서, 새로운 `DetectionInfo` 메시지를 포함하도록 구조를 변경합니다.

- **수정할 파일:** `shopee_interfaces/msg/PickeeDetectedProduct.msg`
- **수정 후 내용:**
  ```
  # 감지된 상품 하나의 최종 정보
  int32 product_id
  int32 bbox_number  # 화면에 표시될 BBox 번호 (유지)
  float32 confidence
  shopee_interfaces/DetectionInfo detection_info
  ```
