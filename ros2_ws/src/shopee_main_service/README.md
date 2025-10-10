# Shopee Main Service

ROS 2 패키지로 구현된 Shopee 중앙 백엔드 서비스 골격입니다.  
`shopee_interfaces` 메시지를 사용하여 Pickee/Packee 로봇과 통신하고, TCP API를 통해 App과 연결됩니다.

## 구조
- `APIController`: TCP 서버 진입점, 메시지 라우팅
- `UserService`, `ProductService`, `OrderService`: 도메인 로직 스켈레톤
- `RobotCoordinator`: ROS 2 노드, Pickee/Packee 서비스/토픽 연동
- `EventBus`, `DatabaseManager`, `LLMClient`: 내부 헬퍼
- `main_service_node.py`: 모든 모듈을 조립해 실행

## 실행 (개발용)
```bash
cd /home/jinhyuk2me/dev_ws/Shopee/ros2_ws
colcon build --packages-select shopee_interfaces shopee_main_service
source install/setup.bash
ros2 run shopee_main_service main_service_node
```

현재는 스켈레톤이므로 실제 DB/ROS 연동 대신 로그만 출력합니다.  
주요 핸들러/서비스를 구현하면서 설계 문서를 따라 기능을 확장하세요.
