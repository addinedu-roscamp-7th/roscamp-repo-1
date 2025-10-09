# 📡 Interface Specification: App ↔ Main Service (UDP)

**Component:** App ↔ Main Service
**Port:** UDP 6000

---

## 🔹 공통 규약

### 통신 파라미터
| 항목 | 내용 |
|---|---|
| Port | 6000 |
| Protocol | UDP |
| Data Format | JSON (메타데이터) + Binary (이미지 데이터) |
| Max Packet Size | 1,472 bytes (1,400 bytes data + 72 bytes header) |
| Encoding | UTF-8 (JSON), Binary (Image) |
| Image Format | JPEG |
| Resolution | 640x480 |

### 패킷 구조
**패킷 레이아웃**
`[JSON Header (72 bytes)] + [Binary Image Data (1,400 bytes)]`

**JSON Header 포맷**
```json
{
    "type": "video_frame",
    "frame_id": 12345,            // 프레임 고유 식별자 (0~4294967295)
    "chunk_idx": 0,               // 현재 청크 인덱스 (0부터 시작)
    "total_chunks": 50,           // 전체 청크 개수
    "data_size": 1400,            // 이 패킷의 실제 이미지 데이터 크기 (bytes)
    "timestamp": 1234567890123,
    "width": 640,
    "height": 480,
    "format": "jpeg"
}
```

---

## 🔹 인터페이스 상세 명세

### 이벤트

**IF-UDP-001: 영상 송출**
- **Function:** 영상 송출
- **From:** Main Service
- **To:** App
- **Message Type:** `video_frame`
- **상세 메시지 포맷:**
    ```json
    {
        "type": "video_frame",
        "frame_id": 12345,
        "chunk_idx": 0,
        "total_chunks": 50,
        "data_size": 1400,
        "timestamp": 1234567890123,
        "width": 640,
        "height": 480,
        "format": "jpeg"
    }
    ```
    `+ Binary Data (max 1,400 bytes)`
- **비고:** 640x480 JPEG 분할 전송
