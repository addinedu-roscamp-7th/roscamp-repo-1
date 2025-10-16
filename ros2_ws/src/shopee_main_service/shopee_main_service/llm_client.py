"""
LLM 서비스 클라이언트

자연어 처리를 위한 LLM 서버와 HTTP 통신합니다.
- 상품 검색 쿼리 생성
- 음성 명령 의도 파악
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from .config import settings

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

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        HTTP 요청을 재시도하며 수행한다.
        """
        attempts = max(1, settings.LLM_MAX_RETRIES)
        delay = max(0.0, settings.LLM_RETRY_BACKOFF)
        last_error: Optional[Exception] = None
        method_upper = method.upper()

        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient() as client:
                    if method_upper == 'GET':
                        response = await client.get(
                            endpoint,
                            params=payload,
                            timeout=self._timeout,
                        )
                    else:
                        response = await client.post(
                            endpoint,
                            json=payload,
                            timeout=self._timeout,
                        )
                    response.raise_for_status()
                    if response.headers.get('content-length') == '0':
                        return {}
                    return response.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                logger.warning(
                    'LLM request failed (attempt %d/%d) to %s %s: %s',
                    attempt,
                    attempts,
                    method_upper,
                    endpoint,
                    exc,
                )
                if attempt == attempts:
                    break
                if delay > 0:
                    await asyncio.sleep(delay)
                    delay *= 2
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.exception('Unexpected error in LLMClient: %s', exc)
                break

        if isinstance(last_error, httpx.HTTPStatusError):
            logger.error(
                'LLM service returned error status %d: %s',
                last_error.response.status_code,
                last_error.response.text,
            )
        elif isinstance(last_error, httpx.RequestError):
            logger.error(
                'Failed to connect to LLM service at %s: %s',
                last_error.request.url,
                last_error,
            )
        elif last_error:
            logger.error('LLMClient request failed: %s', last_error)
        return None

    async def generate_search_query(self, text: str) -> Optional[str]:
        """
        검색어 → SQL 쿼리 생성
        
        Args:
            text: 자연어 검색어 (예: "비건 사과")
            
        Returns:
            str: LLM이 생성한 SQL 쿼리 문자열 또는 None (실패 시)
        """
        endpoint = f'{self._base_url}/llm/search_query'
        payload = {'text': text}

        logger.debug('LLM search_query request to %s with text=%r', endpoint, text)

        data = await self._request_with_retry('GET', endpoint, payload)
        if not data:
            return None
        sql_query = data.get('sql_query')
        if sql_query:
            logger.info('LLM generated SQL: %s', sql_query)
            return sql_query
        logger.warning("LLM response does not contain 'sql_query'")
        return None

    async def extract_bbox_number(self, text: str) -> Optional[int]:
        """자연어 문장에서 bbox 번호를 추출한다."""
        endpoint = f'{self._base_url}/llm/bbox'
        payload = {'text': text}

        logger.debug('LLM bbox request to %s with text=%r', endpoint, text)

        data = await self._request_with_retry('GET', endpoint, payload)
        if not data:
            return None

        bbox_value = data.get('bbox')
        if bbox_value is None:
            logger.debug('LLM bbox response missing bbox field: %s', data)
            return None

        try:
            return int(bbox_value)
        except (TypeError, ValueError):
            logger.warning('LLM bbox response has invalid value: %s', bbox_value)
            return None

    async def detect_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """자연어 문장의 의도와 엔티티를 추출한다."""
        endpoint = f'{self._base_url}/llm/intent_detection'
        payload = {'text': text}

        logger.debug('LLM intent_detection request to %s with text=%r', endpoint, text)

        data = await self._request_with_retry('GET', endpoint, payload)
        if not data or not data.get('intent'):
            logger.warning('LLM intent response missing intent: %s', data)
            return None
        return data
