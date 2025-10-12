#!/usr/bin/env python3
"""
Mock LLM Server 실행 스크립트
"""
import sys
import logging

try:
    from aiohttp import web
except ImportError:
    print("Error: aiohttp is not installed.")
    print("Install it with: pip install aiohttp")
    sys.exit(1)

logger = logging.getLogger("mock_llm_server")


class MockLLMServer:
    """Mock LLM HTTP 서버"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """API 라우트 설정"""
        self.app.router.add_post("/llm_service/search_query", self.handle_search_query)
        self.app.router.add_post("/llm_service/intent_detection", self.handle_detect_intent)

    async def handle_search_query(self, request):
        """상품 검색 쿼리 생성"""
        try:
            data = await request.json()
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
            data = await request.json()
            text = data.get("text", "")
            logger.info(f"[MOCK LLM] Intent detection: {text}")

            intent_data = self._analyze_mock_intent(text)
            logger.info(f"[MOCK LLM] Detected: {intent_data}")

            return web.json_response(intent_data)
        except Exception as e:
            logger.error(f"Error: {e}")
            return web.json_response({"error": str(e)}, status=500)

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

    def _analyze_mock_intent(self, text: str):
        """간단한 인텐트 분석"""
        text_lower = text.lower()

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
  POST http://localhost:8000/llm_service/search_query
  POST http://localhost:8000/llm_service/intent_detection

Press Ctrl+C to stop
""")

    server = MockLLMServer(host="0.0.0.0", port=8000)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nMock LLM Server stopped")

if __name__ == "__main__":
    main()
