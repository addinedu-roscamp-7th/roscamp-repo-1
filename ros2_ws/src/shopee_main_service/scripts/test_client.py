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


class TestClient:
    """Main Service TCP 테스트 클라이언트"""

    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        """서버에 연결"""
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print(f"✓ Connected to {self.host}:{self.port}")

    async def send_request(self, msg_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """요청 전송 및 응답 수신"""
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

        # 응답 수신
        response_line = await self.reader.readline()
        response = json.loads(response_line.decode())

        print(f"← Received: {response['type']}")
        print(f"  Result: {response['result']}")
        if response.get('message'):
            print(f"  Message: {response['message']}")
        if response.get('data'):
            print(f"  Data: {json.dumps(response['data'], ensure_ascii=False, indent=2)}")

        return response

    async def close(self):
        """연결 종료"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        print("\n✓ Connection closed")

    async def test_full_workflow(self):
        """전체 워크플로우 테스트"""
        print("\n" + "="*60)
        print("Starting Full Workflow Test")
        print("="*60)

        # 1. 로그인
        print("\n[1] Testing Login...")
        await self.send_request("user_login", {
            "user_id": "admin",
            "password": "admin123"
        })
        await asyncio.sleep(0.5)

        # 2. 상품 검색
        print("\n[2] Testing Product Search...")
        await self.send_request("product_search", {
            "query": "비건 사과"
        })
        await asyncio.sleep(0.5)

        # 3. 주문 생성
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
        print("\n[4] Testing Video Stream Start...")
        await self.send_request("video_stream_start", {
            "robot_id": robot_id,
            "user_id": "admin",
            "user_type": "customer"
        })
        await asyncio.sleep(0.5)

        # 5. 상품 선택 (로봇이 상품 인식 완료 후)
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
        print("\n[6] Testing Shopping End...")
        await self.send_request("shopping_end", {
            "user_id": "admin",
            "order_id": order_id,
            "robot_id": robot_id
        })
        await asyncio.sleep(2)  # 포장 대기

        # 7. 영상 스트림 중지
        print("\n[7] Testing Video Stream Stop...")
        await self.send_request("video_stream_stop", {
            "robot_id": robot_id,
            "user_id": "admin",
            "user_type": "customer"
        })
        await asyncio.sleep(0.5)

        # 8. 재고 검색
        print("\n[8] Testing Inventory Search...")
        await self.send_request("inventory_search", {
            "name": "사과"
        })
        await asyncio.sleep(0.5)

        # 9. 로봇 히스토리 검색
        print("\n[9] Testing Robot History Search...")
        await self.send_request("robot_history_search", {
            "robot_id": robot_id
        })
        await asyncio.sleep(0.5)

        # 10. 음성 명령 테스트
        print("\n[10] Testing Voice Command...")
        await self.send_request("voice_command", {
            "text": "사과 한 개 가져다줘"
        })

        print("\n" + "="*60)
        print("Full Workflow Test Completed!")
        print("="*60)

    async def test_inventory_management(self):
        """재고 관리 기능 테스트"""
        print("\n" + "="*60)
        print("Testing Inventory Management")
        print("="*60)

        # 재고 추가
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
        print("\n[2] Searching Product...")
        await self.send_request("inventory_search", {
            "product_id": 9999
        })
        await asyncio.sleep(0.3)

        # 재고 수정
        print("\n[3] Updating Product...")
        await self.send_request("inventory_update", {
            "product_id": 9999,
            "price": 4500,
            "quantity": 150
        })
        await asyncio.sleep(0.3)

        # 재고 삭제
        print("\n[4] Deleting Product...")
        await self.send_request("inventory_delete", {
            "product_id": 9999
        })

        print("\n" + "="*60)
        print("Inventory Management Test Completed!")
        print("="*60)


async def main():
    """메인 함수"""
    client = TestClient(host="localhost", port=5000)

    try:
        await client.connect()

        if len(sys.argv) > 1 and sys.argv[1] == "inventory":
            await client.test_inventory_management()
        else:
            await client.test_full_workflow()

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
  python3 test_client.py              # Test full workflow
  python3 test_client.py inventory    # Test inventory management

Before running:
  1. Start Mock LLM Server:  python3 -m shopee_main_service.mock_llm_server
  2. Start Mock Robot Node:  ros2 run shopee_main_service mock_robot_node
  3. Start Main Service:     ros2 run shopee_main_service main_service_node
""")

    asyncio.run(main())
