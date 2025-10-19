#!/usr/bin/env python3
"""
Mock LLM Server 실행 스크립트
"""
import sys
import logging
from typing import Optional

try:
    from aiohttp import web
except ImportError:
    print("Error: aiohttp is not installed.")
    print("Install it with: pip install aiohttp")
    sys.exit(1)

logger = logging.getLogger("mock_llm_server")


class MockLLMServer:
    """Mock LLM HTTP 서버"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5001):
        self.host = host
        self.port = port
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """API 라우트 설정"""
        self.app.router.add_get("/llm/search_query", self.handle_search_query)
        self.app.router.add_get("/llm/intent_detection", self.handle_detect_intent)
        self.app.router.add_get("/llm/bbox", self.handle_bbox_extract)

    async def handle_search_query(self, request):
        """상품 검색 쿼리 생성"""
        try:
            data = await self._get_payload(request)
            text = data.get("text", "")
            logger.info(f"[MOCK LLM] Search query: {text}")

            sql_query = self._generate_mock_search_query(text)
            logger.info(f"[MOCK LLM] Generated: {sql_query}")

            return web.json_response({"sql_query": sql_query})
        except Exception as e:
            logger.error(f"Error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_detect_intent(self, request):
        """음성 명령 인텐트 분석"""
        try:
            data = await self._get_payload(request)
            text = data.get("text", "")
            logger.info(f"[MOCK LLM] Intent detection: {text}")

            intent_data = self._analyze_mock_intent(text)
            logger.info(f"[MOCK LLM] Detected: {intent_data}")

            return web.json_response(intent_data)
        except Exception as e:
            logger.error(f"Error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_bbox_extract(self, request):
        """bbox 번호 추출"""
        try:
            data = await self._get_payload(request)
            text = data.get("text", "")
            logger.info(f"[MOCK LLM] BBox extract: {text}")

            bbox_number = self._extract_bbox_number(text)
            return web.json_response({"bbox": bbox_number} if bbox_number is not None else {})
        except Exception as e:
            logger.error(f"Error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _get_payload(self, request):
        """GET 요청에서도 JSON 혹은 쿼리 파라미터를 파싱한다."""
        if request.method.upper() == "GET":
            query_data = dict(request.rel_url.query)
            if query_data:
                return query_data
        try:
            return await request.json()
        except Exception:
            return {}

    def _generate_mock_search_query(self, query: str) -> str:
        """간단한 검색 쿼리 생성"""
        conditions = []
        keywords = []
        vegan = False
        category = None

        words = query.split()
        for word in words:
            if word in ["비건", "채식", "vegan"]:
                vegan = True
            elif word in ["과일", "채소", "음료", "간식", "유제품"]:
                category = word
            else:
                keywords.append(word)

        if keywords:
            keyword_conditions = " OR ".join([f"name LIKE '%{kw}%'" for kw in keywords])
            conditions.append(f"({keyword_conditions})")

        if vegan:
            conditions.append("is_vegan_friendly = true")

        if category:
            conditions.append(f"category = '{category}'")

        return " AND ".join(conditions) if conditions else "1=1"

    def _extract_bbox_number(self, text: str) -> Optional[int]:
        """단순히 숫자를 파싱하여 bbox 번호로 사용한다."""
        for token in text.replace("번", " ").split():
            cleaned = "".join(ch for ch in token if ch.isdigit())
            if cleaned:
                try:
                    return int(cleaned)
                except ValueError:
                    continue
        return None

    def _analyze_mock_intent(self, text: str):
        """간단한 인텐트 분석"""
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ["이동", "move", "가자", "가줘"]):
            place_name = None
            for word in text.replace(",", " ").split():
                if word.endswith("로") or word.endswith("에"):
                    place_name = word.rstrip("로").rstrip("에")
                    break
            return {
                "intent": "Move_place",
                "entities": {
                    "place_name": place_name or "destination",
                    "action": "move",
                },
            }

        if any(word in text_lower for word in ["가져다", "찾아", "주세요", "원해", "사고싶"]):
            product_name = None
            for word in text.split():
                if word not in ["가져다줘", "한", "개", "주세요", "찾아줘"]:
                    product_name = word.replace("를", "").replace("을", "")
                    break

            quantity = 1
            if "두" in text or "2" in text:
                quantity = 2

            return {
                "intent": "fetch_product",
                "entities": {
                    "product_name": product_name or "unknown",
                    "quantity": quantity
                }
            }

        return {"intent": "unknown", "entities": {}}

    def run(self):
        """서버 실행"""
        logger.info(f"Starting Mock LLM Server on {self.host}:{self.port}")
        web.run_app(self.app, host=self.host, port=self.port)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("""
╔══════════════════════════════════════════════════════════╗
║             Mock LLM Server Starting                     ║
╚══════════════════════════════════════════════════════════╝

Endpoints:
  GET http://localhost:5001/llm/search_query
  GET http://localhost:5001/llm/intent_detection
  GET http://localhost:5001/llm/bbox

Press Ctrl+C to stop
""")

    server = MockLLMServer(host="0.0.0.0", port=5001)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nMock LLM Server stopped")

if __name__ == "__main__":
    main()
