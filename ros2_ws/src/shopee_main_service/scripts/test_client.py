#!/usr/bin/env python3
"""
Test Client - Main Service 테스트 클라이언트

Mock 환경에서 Main Service의 모든 기능을 테스트하는 CLI 클라이언트입니다.
"""
import asyncio
import json
import socket
import sys
from typing import Dict, Any


class MainServiceClient:
    """Main Service TCP 테스트 클라이언트"""

    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self.async_notifications = []  # 비동기 알림 저장

    async def connect(self):
        """서버에 연결"""
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print(f"✓ Connected to {self.host}:{self.port}")

    def _is_async_notification(self, msg_type: str) -> bool:
        """비동기 알림인지 확인"""
        async_types = [
            'robot_moving_notification',
            'robot_arrived_notification',
            'product_selection_start',
            'cart_update_notification',
            'work_info_notification',
            'packing_info_notification'
        ]
        return msg_type in async_types

    def _print_notification(self, response: Dict[str, Any]):
        """비동기 알림 출력"""
        print(f"\n  📢 [Async Notification] {response['type']}")
        if response.get('message'):
            print(f"     Message: {response['message']}")
        if response.get('data'):
            print(f"     Data: {json.dumps(response['data'], ensure_ascii=False, indent=2)}")

    async def send_request(self, msg_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """요청 전송 및 응답 수신 (비동기 알림 처리 포함)"""
        request = {
            "type": msg_type,
            "data": data
        }

        # 요청 전송
        request_json = json.dumps(request) + "\n"
        self.writer.write(request_json.encode())
        await self.writer.drain()

        print(f"\n→ Sent: {msg_type}")
        print(f"  Data: {json.dumps(data, ensure_ascii=False)}")

        # 응답 수신 - 비동기 알림 건너뛰고 올바른 응답 찾기
        expected_response = msg_type + "_response"
        max_attempts = 20  # 최대 20개 메시지까지 읽기

        for attempt in range(max_attempts):
            try:
                response_line = await asyncio.wait_for(self.reader.readline(), timeout=5.0)
                if not response_line:
                    print("  ⚠️ Connection closed by server")
                    break

                response = json.loads(response_line.decode())

                # 비동기 알림은 저장하고 계속 읽기
                if self._is_async_notification(response['type']):
                    self._print_notification(response)
                    self.async_notifications.append(response)
                    continue

                # 기대하는 응답이면 출력 후 반환
                if response['type'] == expected_response:
                    print(f"← Received: {response['type']}")
                    print(f"  Result: {response['result']}")
                    if response.get('message'):
                        print(f"  Message: {response['message']}")
                    if response.get('data'):
                        print(f"  Data: {json.dumps(response['data'], ensure_ascii=False, indent=2)}")
                    return response

                # 예상치 못한 응답
                print(f"  ⚠️ Unexpected response: {response['type']} (expected: {expected_response})")
                return response

            except asyncio.TimeoutError:
                print(f"  ⚠️ Timeout waiting for {expected_response}")
                break
            except Exception as e:
                print(f"  ⚠️ Error reading response: {e}")
                break

        print(f"  ❌ Failed to receive {expected_response}")
        return {"type": "error", "result": False, "message": "No response received"}

    async def close(self):
        """연결 종료"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        print("\n✓ Connection closed")

    def wait_for_user(self, step_num: int, description: str):
        """사용자 입력 대기 (인터랙티브 모드용)"""
        input(f"\n[{step_num}] {description}\n→ Press Enter to continue...")

    async def test_full_workflow(self, interactive: bool = False):
        """전체 워크플로우 테스트"""
        print("\n" + "="*60)
        print("Starting Full Workflow Test")
        if interactive:
            print("(Interactive Mode - Press Enter to proceed each step)")
        print("="*60)

        # 1. 로그인
        if interactive:
            self.wait_for_user(1, "Testing Login...")
        else:
            print("\n[1] Testing Login...")
        await self.send_request("user_login", {
            "user_id": "admin",
            "password": "admin123"
        })
        await asyncio.sleep(0.5)

        # 2. 상품 검색
        if interactive:
            self.wait_for_user(2, "Testing Product Search...")
        else:
            print("\n[2] Testing Product Search...")
        await self.send_request("product_search", {
            "query": "비건 사과"
        })
        await asyncio.sleep(0.5)

        # 3. 주문 생성
        if interactive:
            self.wait_for_user(3, "Testing Order Creation...")
        else:
            print("\n[3] Testing Order Creation...")
        order_response = await self.send_request("order_create", {
            "user_id": "admin",
            "cart_items": [
                {"product_id": 1, "quantity": 2},
                {"product_id": 2, "quantity": 1}
            ]
        })
        await asyncio.sleep(1)  # 로봇 이동 대기

        if not order_response['result']:
            print("✗ Order creation failed, stopping test")
            return

        order_id = order_response['data']['order_id']
        robot_id = order_response['data']['robot_id']

        # 4. 영상 스트림 시작
        if interactive:
            self.wait_for_user(4, "Testing Video Stream Start...")
        else:
            print("\n[4] Testing Video Stream Start...")
        await self.send_request("video_stream_start", {
            "robot_id": robot_id,
            "user_id": "admin",
            "user_type": "customer"
        })
        await asyncio.sleep(0.5)

        # 5. 상품 선택 (로봇이 상품 인식 완료 후)
        if interactive:
            self.wait_for_user(5, "Testing Product Selection...")
        else:
            print("\n[5] Testing Product Selection...")
        await asyncio.sleep(1)  # 상품 인식 대기
        await self.send_request("product_selection", {
            "order_id": order_id,
            "robot_id": robot_id,
            "bbox_number": 1,
            "product_id": 1
        })
        await asyncio.sleep(1)  # 선택 처리 대기

        # 6. 쇼핑 종료
        if interactive:
            self.wait_for_user(6, "Testing Shopping End...")
        else:
            print("\n[6] Testing Shopping End...")
        await self.send_request("shopping_end", {
            "user_id": "admin",
            "order_id": order_id,
            "robot_id": robot_id
        })
        await asyncio.sleep(2)  # 포장 대기

        # 7. 영상 스트림 중지
        if interactive:
            self.wait_for_user(7, "Testing Video Stream Stop...")
        else:
            print("\n[7] Testing Video Stream Stop...")
        await self.send_request("video_stream_stop", {
            "robot_id": robot_id,
            "user_id": "admin",
            "user_type": "customer"
        })
        await asyncio.sleep(0.5)

        # 8. 재고 검색
        if interactive:
            self.wait_for_user(8, "Testing Inventory Search...")
        else:
            print("\n[8] Testing Inventory Search...")
        await self.send_request("inventory_search", {
            "name": "사과"
        })
        await asyncio.sleep(0.5)

        # 9. 로봇 히스토리 검색
        if interactive:
            self.wait_for_user(9, "Testing Robot History Search...")
        else:
            print("\n[9] Testing Robot History Search...")
        await self.send_request("robot_history_search", {
            "robot_id": robot_id
        })
        await asyncio.sleep(0.5)

        # 10. 음성 명령 테스트
        if interactive:
            self.wait_for_user(10, "Testing Voice Command...")
        else:
            print("\n[10] Testing Voice Command...")
        # 포장 완료 대기
        print("\n⏳ Waiting for packaging completion...")
        await asyncio.sleep(2)

        print("\n" + "="*60)
        print("Full Workflow Test Completed!")
        print("="*60)

        # 비동기 알림 요약
        print(f"\n📊 Async Notifications Received: {len(self.async_notifications)}")
        notification_types = {}
        for notif in self.async_notifications:
            notif_type = notif.get('type', 'unknown')
            notification_types[notif_type] = notification_types.get(notif_type, 0) + 1

        if notification_types:
            print("\nNotification Summary:")
            for notif_type, count in sorted(notification_types.items()):
                print(f"  - {notif_type}: {count}")

        # 워크플로우 체크리스트
        expected_notifications = [
            'robot_moving_notification',
            'robot_arrived_notification',
            'product_selection_start',
            'cart_update_notification',
        ]

        print("\n✅ Workflow Checklist:")
        for expected in expected_notifications:
            if expected in notification_types:
                print(f"  ✓ {expected}")
            else:
                print(f"  ✗ {expected} (MISSING!)")

        print("\n" + "="*60)

    async def test_inventory_management(self, interactive: bool = False):
        """재고 관리 기능 테스트"""
        print("\n" + "="*60)
        print("Testing Inventory Management")
        if interactive:
            print("(Interactive Mode - Press Enter to proceed each step)")
        print("="*60)

        # 재고 추가
        if interactive:
            self.wait_for_user(1, "Creating Product...")
        else:
            print("\n[1] Creating Product...")
        await self.send_request("inventory_create", {
            "product_id": 9999,
            "barcode": "TEST999",
            "name": "테스트 상품",
            "quantity": 100,
            "price": 5000,
            "discount_rate": 10,
            "category": "테스트",
            "allergy_info_id": 1,
            "is_vegan_friendly": True,
            "section_id": 1
        })
        await asyncio.sleep(0.3)

        # 재고 검색
        if interactive:
            self.wait_for_user(2, "Searching Product...")
        else:
            print("\n[2] Searching Product...")
        await self.send_request("inventory_search", {
            "product_id": 9999
        })
        await asyncio.sleep(0.3)

        # 재고 수정
        if interactive:
            self.wait_for_user(3, "Updating Product...")
        else:
            print("\n[3] Updating Product...")
        await self.send_request("inventory_update", {
            "product_id": 9999,
            "price": 4500,
            "quantity": 150
        })
        await asyncio.sleep(0.3)

        # 재고 삭제
        if interactive:
            self.wait_for_user(4, "Deleting Product...")
        else:
            print("\n[4] Deleting Product...")
        await self.send_request("inventory_delete", {
            "product_id": 9999
        })

        print("\n" + "="*60)
        print("Inventory Management Test Completed!")
        print("="*60)


async def main():
    """메인 함수"""
    client = MainServiceClient(host="localhost", port=5000)

    # 인터랙티브 모드 확인
    interactive = "-i" in sys.argv or "--interactive" in sys.argv

    try:
        await client.connect()

        if "inventory" in sys.argv:
            await client.test_inventory_management(interactive=interactive)
        else:
            await client.test_full_workflow(interactive=interactive)

    except ConnectionRefusedError:
        print("\n✗ Error: Could not connect to Main Service")
        print("  Make sure Main Service is running on localhost:5000")
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║         Shopee Main Service Test Client                 ║
╚══════════════════════════════════════════════════════════╝

Usage:
  python3 test_client.py                    # Test full workflow (auto)
  python3 test_client.py -i                 # Test full workflow (interactive)
  python3 test_client.py inventory          # Test inventory management (auto)
  python3 test_client.py inventory -i       # Test inventory management (interactive)

Options:
  -i, --interactive   Run in interactive mode (press Enter for each step)

Before running:
  1. Start Mock LLM Server:  python3 -m shopee_main_service.mock_llm_server
  2. Start Mock Robot Node:  ros2 run shopee_main_service mock_robot_node
  3. Start Main Service:     ros2 run shopee_main_service main_service_node
""")

    asyncio.run(main())
