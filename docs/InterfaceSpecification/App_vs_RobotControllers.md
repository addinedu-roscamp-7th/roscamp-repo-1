Shopee App (Admin Dashboard) = 쇼피 관리자 앱
Pickee Main Controller = 피키 메인 컨트롤러
Packee Main Controller = 패키 메인 컨트롤러

## 통신 개요
- **프로토콜**: ROS 2 Topic (rosbridge WebSocket 구독)
- **전송 방향**: 로봇 컨트롤러 → Shopee App (읽기 전용)
- **목적**: 관리자 대시보드에서 실시간 로봇 상태 / 위치 / 작업 상황을 표시
- **갱신 주기**: 기본 30초 폴링 + 이벤트 발생 시 즉시 Push (토픽 발행 시점 기준)

## Topic 인터페이스

### Pickee 로봇 상태 스트림
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeRobotStatus.msg`
- **Topic**: `/pickee/robot_status`
- **ROS 타입**: `shopee_interfaces/msg/PickeeRobotStatus`
- **From**: Pickee Main Controller
- **To**: Shopee App (Admin)

### Packee 로봇 상태 스트림
> **ROS2 Interface:** `shopee_interfaces/msg/PackeeRobotStatus.msg`
- **Topic**: `/packee/robot_status`
- **ROS 타입**: `shopee_interfaces/msg/PackeeRobotStatus`
- **From**: Packee Main Controller
- **To**: Shopee App (Admin)

### Pickee 위치 확인 (선택)
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeMobilePose.msg`
- **Topic**: `/pickee/mobile/pose`
- **ROS 타입**: `shopee_interfaces/msg/PickeeMobilePose`
- **From**: Pickee Mobile Controller
- **To**: Shopee App (Admin)

### 이벤트 트리거 정리
| 시퀀스 ID | 사용 토픽 | 설명 |
|---|---|---|
| `SC_05_1_1` | `/pickee/robot_status`, `/packee/robot_status` | 대시보드 메인 카드/리스트 갱신 |
| `SC_05_1_2` | `/pickee/robot_status`, `/pickee/mobile/pose` | 위치 지도 갱신 |
| `SC_05_1_4` | `/pickee/robot_status`, `/packee/robot_status` | 로봇 상태 패널 |

## 보안 및 운영 고려사항
- rosbridge WebSocket 연결은 관리자 인증 이후에만 활성화.
- ROS Topic은 읽기 전용(subscribe)으로 제한; 관리자 앱에서 Publish 금지.
- 네트워크 대역폭 확보를 위해 1초 이하 간격의 연속 발행 시 샘플링(Throttle) 적용 권장.
- 필요 시 Shopee Main Service가 필터링/캐싱한 후 앱에 재전송하도록 중간 게이트웨이 구성 가능.
