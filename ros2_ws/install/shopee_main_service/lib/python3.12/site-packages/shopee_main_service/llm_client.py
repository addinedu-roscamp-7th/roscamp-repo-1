"""
LLM 서비스 클라이언트

자연어 처리를 위한 LLM 서버와 HTTP 통신합니다.
- 상품 검색 쿼리 생성
- 음성 명령 의도 파악
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM HTTP 클라이언트
    
    LLM 서버와 통신하여 자연어 처리 기능을 제공합니다.
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
            str: LLM이 생성한 SQL 쿼리 문자열 또는 None (실패 시)
        """
        endpoint = f"{self._base_url}/llm_service/search_query"
        payload = {"text": text}
        
        logger.debug("LLM search_query request to %s with text='%s'", endpoint, text)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    timeout=self._timeout
                )
                response.raise_for_status()  # 2xx 이외의 상태 코드에 대해 예외 발생
                
                data = response.json()
                sql_query = data.get("sql_query")
                
                if sql_query:
                    logger.info("LLM generated SQL: %s", sql_query)
                    return sql_query
                else:
                    logger.warning("LLM response does not contain 'sql_query'")
                    return None

        except httpx.HTTPStatusError as e:
            logger.error("LLM service returned error status %d: %s", e.response.status_code, e.response.text)
            return None
        except httpx.RequestError as e:
            logger.error("Failed to connect to LLM service at %s: %s", e.request.url, e)
            return None
        except Exception as e:
            logger.exception("An unexpected error occurred in LLMClient: %s", e)
            return None
