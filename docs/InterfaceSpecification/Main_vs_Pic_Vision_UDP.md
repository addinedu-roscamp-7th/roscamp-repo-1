# 📡 Interface Specification: Pickee Vision → Shopee Main (UDP)

**Components:** Pickee Vision AI Service → Shopee Main Service  
**Purpose:** 관리자 스트리밍 요청이 활성화된 동안 Vision에서 추출한 영상을 Main이 수신하여 App으로 중계

---

## 🔹 공통 규약

| 항목 | 내용 |
|---|---|
| Port | 6000 |
| Protocol | UDP |
| Data Format | JSON Header + Binary (JPEG) |
| Max Packet Size | 1,472 bytes (헤더 72 + 데이터 1,400) |
| Encoding | UTF-8 (JSON), Binary (Image) |
| 이미지 사양 | 640×480, JPEG |

> Main Service는 동일 포트(6000)로 Vision에서 들어오는 프레임을 수신하고, 검증 후 동일 메시지 구조로 App에 전달한다.

---

## 🔹 패킷 구조

`[JSON Header (72 bytes)] + [Binary Image Data (≤ 1,400 bytes)]`

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

- `frame_id`: Vision이 부여한 프레임 고유 ID (0~4294967295)
- `chunk_idx`: 현재 청크 인덱스 (0부터 시작)
- `total_chunks`: 한 프레임을 구성하는 총 청크 수
- `data_size`: 본 패킷의 실제 이미지 데이터 바이트 수
- `timestamp`: Vision에서 송출한 시각(ms)

---

## 🔹 이벤트

### IF-UDP-050: Vision 영상 프레임 송출
- **From:** Pickee Vision AI Service  
- **To:** Shopee Main Service  
- **Message Type:** `video_frame`
- **설명:** Vision은 `/pickee/video_stream/start` 명령을 수락한 동안 해당 포맷으로 프레임을 지속 전송한다.
- **수신 처리:** Main Service는 프레임 순서를 복원하여 App으로 재전송(`App_vs_Main_UDP.md` 참조)하며, 에러 발생 시 `/pickee/video_stream/stop`을 호출하거나 관리자에게 오류 메시지를 전달한다.

---

## 🔹 운영 참고
- Vision ↔ Main UDP 통신은 내부 네트워크에서만 허용하며, 방화벽에서 포트 6000/UDP에 대한 허용 규칙이 필요하다.
- 패킷 손실 감지를 위해 Main은 `chunk_idx/total_chunks` 기반 재조립 시 누락된 청크를 확인하고 요청을 종료할 수 있다.
- 고해상도 스트리밍이 필요한 경우 향후 H.264 RTP 등 별도 프로토콜을 정의해야 한다.
