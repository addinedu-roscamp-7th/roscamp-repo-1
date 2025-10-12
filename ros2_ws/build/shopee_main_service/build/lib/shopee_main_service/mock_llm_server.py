#!/usr/bin/env python3
"""
Mock LLM Server - LLM 시뮬레이터

실제 LLM 없이 Main Service를 테스트하기 위한 Mock LLM HTTP 서버입니다.
상품 검색 쿼리 생성과 음성 명령 인텐트 분석을 시뮬레이션합니다.
"""
import sys
import asyncio
import json
import logging
from typing import Dict

try:
    from aiohttp import web
except ImportError:
    print("Error: aiohttp is not installed. Install it with: pip install aiohttp")
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
        self.app.router.add_post("/search_query", self.handle_search_query)
        self.app.router.add_post("/detect_intent", self.handle_detect_intent)

    async def handle_search_query(self, request: web.Request) -> web.Response:
        """
        상품 검색 쿼리 생성 시뮬레이션

        Request: {"query": "비건 사과"}
        Response: {"where_clause": "name LIKE '%사과%' AND is_vegan_friendly = true"}
        """
        try:
            data = await request.json()
            query = data.get("query", "")

            logger.info(f"[MOCK LLM] Search query request: {query}")

            # 간단한 규칙 기반 쿼리 생성
            where_clause = self._generate_mock_search_query(query)

            response_data = {"where_clause": where_clause}
            logger.info(f"[MOCK LLM] Generated query: {where_clause}")

            return web.json_response(response_data)

        except Exception as e:
            logger.error(f"[MOCK LLM] Error in search_query: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_detect_intent(self, request: web.Request) -> web.Response:
        """
        음성 명령 인텐트 분석 시뮬레이션

        Request: {"text": "사과 한 개 가져다줘"}
        Response: {
            "intent": "fetch_product",
            "entities": {
                "product_name": "사과",
                "quantity": 1
            }
        }
        """
        try:
            data = await request.json()
            text = data.get("text", "")

            logger.info(f"[MOCK LLM] Intent detection request: {text}")

            # 간단한 규칙 기반 인텐트 분석
            intent_data = self._analyze_mock_intent(text)

            logger.info(f"[MOCK LLM] Detected intent: {intent_data}")

            return web.json_response(intent_data)

        except Exception as e:
            logger.error(f"[MOCK LLM] Error in detect_intent: {e}")
            return web.json_response({"error": str(e)}, status=500)

    def _generate_mock_search_query(self, query: str) -> str:
        """간단한 규칙 기반 검색 쿼리 생성"""
        conditions = []

        # 상품명 검색
        if query:
            # 키워드 추출 (비건, 알레르기 등 제외)
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

            # 키워드 조건
            if keywords:
                keyword_conditions = " OR ".join([f"name LIKE '%{kw}%'" for kw in keywords])
                conditions.append(f"({keyword_conditions})")

            # 비건 조건
            if vegan:
                conditions.append("is_vegan_friendly = true")

            # 카테고리 조건
            if category:
                conditions.append(f"category = '{category}'")

        # 조건이 없으면 전체 검색
        if not conditions:
            return "1=1"

        return " AND ".join(conditions)

    def _analyze_mock_intent(self, text: str) -> Dict:
        """간단한 규칙 기반 인텐트 분석"""
        text_lower = text.lower()

        # 상품 가져오기 인텐트
        if any(word in text_lower for word in ["가져다", "찾아", "주세요", "원해", "사고싶"]):
            # 상품명 추출 (매우 간단한 방식)
            product_name = None
            for word in text.split():
                if word not in ["가져다줘", "한", "개", "주세요", "찾아줘"]:
                    product_name = word.replace("를", "").replace("을", "")
                    break

            # 수량 추출
            quantity = 1
            if "두" in text or "2" in text:
                quantity = 2
            elif "세" in text or "3" in text:
                quantity = 3

            return {
                "intent": "fetch_product",
                "entities": {
                    "product_name": product_name or "unknown",
                    "quantity": quantity
                }
            }

        # 도움 요청 인텐트
        elif any(word in text_lower for word in ["도와", "help", "헬프"]):
            return {
                "intent": "request_help",
                "entities": {}
            }

        # 위치 안내 인텐트
        elif any(word in text_lower for word in ["어디", "위치", "location"]):
            product_name = None
            for word in text.split():
                if word not in ["어디", "있어", "위치", "어디있어요"]:
                    product_name = word.replace("는", "").replace("은", "")
                    break

            return {
                "intent": "find_location",
                "entities": {
                    "product_name": product_name or "unknown"
                }
            }

        # 기본 인텐트
        return {
            "intent": "unknown",
            "entities": {}
        }

    async def start(self):
        """서버 시작"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Mock LLM Server running on http://{self.host}:{self.port}")

    def run(self):
        """서버 실행 (블로킹)"""
        logger.info(f"Starting Mock LLM Server on {self.host}:{self.port}")
        web.run_app(self.app, host=self.host, port=self.port)


def main():
    """Mock LLM Server 메인 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    server = MockLLMServer(host="0.0.0.0", port=8000)
    server.run()


if __name__ == "__main__":
    main()
