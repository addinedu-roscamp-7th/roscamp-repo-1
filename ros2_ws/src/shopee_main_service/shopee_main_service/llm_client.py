"""
LLM 서비스 클라이언트

자연어 처리를 위한 LLM 서버와 HTTP 통신합니다.
- 상품 검색 쿼리 생성
- 음성 명령 의도 파악
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM HTTP 클라이언트
    
    LLM 서버(localhost:8000)와 통신하여 자연어 처리 기능을 제공합니다.
    """

    def __init__(self, base_url: str, timeout: float = 1.5) -> None:
        """
        Args:
            base_url: LLM 서버 URL (예: "http://localhost:8000")
            timeout: HTTP 타임아웃 (초)
        """
        self._base_url = base_url
        self._timeout = timeout

    async def generate_search_query(self, text: str) -> Optional[str]:
        """
        검색어 → SQL 쿼리 생성
        
        Args:
            text: 자연어 검색어 (예: "비건 사과")
            
        Returns:
            str: SQL WHERE 절 또는 None (실패 시)
            
        구현 예정:
            POST /llm_service/search-query
            {
                "query": "비건 사과",
                "schema": {...}  # DB 스키마 정보
            }
            
            응답: {"sql": "WHERE is_vegan_friendly=1 AND name LIKE '%사과%'"}
            
        참고: Main_vs_LLM.md
        """
        logger.debug("LLM search_query request text=%s", text)
        # TODO: integrate httpx client
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         f"{self._base_url}/search-query",
        #         json={"query": text},
        #         timeout=self._timeout
        #     )
        #     return response.json()["sql"]
        return None

    async def detect_intent(self, text: str) -> Dict[str, Any]:
        """
        음성 명령 의도 파악
        
        Args:
            text: STT 결과 텍스트 (예: "재고 보충 모드 켜줘")
            
        Returns:
            dict: 의도 정보 {"intent": "...", "params": {...}}
            
        구현 예정:
            POST /llm_service/intent-detection
            응답: {"intent": "enable_restocking_mode"}
            
        참고: Main_vs_LLM.md
        """
        logger.debug("LLM intent_detection request text=%s", text)
        # TODO: HTTP 요청 구현
        return {}
