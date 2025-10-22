"""
LLM 서비스 클라이언트

자연어 처리를 위한 LLM 서버와 HTTP 통신합니다.
- 상품 검색 쿼리 생성
- 음성 명령 의도 파악
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM HTTP 클라이언트

    LLM 서버와 통신하여 자연어 처리 기능을 제공합니다.
    - 상품 검색 쿼리 생성 (자연어 → SQL)
    - BBox 번호 추출 (음성 명령)
    - 의도 파악 (Intent Detection)

    호출 통계를 자동으로 기록하며, 실패 시 재시도 메커니즘을 제공합니다.
    """

    def __init__(self, base_url: str, timeout: float = 1.5) -> None:
        """
        LLM 클라이언트를 초기화합니다.

        Args:
            base_url: LLM 서버 URL (예: "http://localhost:8000")
            timeout: HTTP 타임아웃 (초). 기본값 1.5초

        사용 예:
            llm_client = LLMClient(base_url="http://localhost:8000", timeout=2.0)
        """
        self._base_url = base_url
        self._timeout = timeout
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, float] = {
            'success_count': 0.0,
            'failure_count': 0.0,
            'fallback_count': 0.0,
            'total_latency_ms': 0.0,
        }

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        HTTP 요청을 재시도하며 수행합니다. (내부 메서드)

        실패 시 지수 백오프(exponential backoff)를 적용하여 재시도합니다.
        모든 재시도가 실패하면 None을 반환합니다.

        Args:
            method: HTTP 메서드 ("GET" 또는 "POST")
            endpoint: 전체 엔드포인트 URL
            payload: 요청 페이로드 (GET은 query params, POST는 JSON body)

        Returns:
            Optional[Dict[str, Any]]: 응답 JSON 또는 None (실패 시)
        """
        attempts = max(1, settings.LLM_MAX_RETRIES)
        delay = max(0.0, settings.LLM_RETRY_BACKOFF)
        last_error: Optional[Exception] = None
        method_upper = method.upper()

        for attempt in range(1, attempts + 1):
            start_time = time.perf_counter()
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
                        self._record_success((time.perf_counter() - start_time) * 1000.0)
                        return {}
                    data = response.json()
                    self._record_success((time.perf_counter() - start_time) * 1000.0)
                    return data
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
        if last_error:
            self._record_failure()
        return None

    async def generate_search_query(self, text: str) -> Optional[str]:
        """
        자연어 검색어를 SQL 쿼리로 변환합니다.

        LLM 서버의 `/llm/search_query` 엔드포인트를 호출하여
        사용자의 자연어 검색어를 데이터베이스 쿼리로 변환합니다.

        Args:
            text: 자연어 검색어 (예: "비건 사과", "견과류 없는 간식")

        Returns:
            Optional[str]: LLM이 생성한 SQL 쿼리 문자열 또는 None (실패 시)

        사용 예:
            sql_query = await llm_client.generate_search_query("비건 사과")
            if sql_query:
                print(f"생성된 쿼리: {sql_query}")
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
        """
        자연어 문장에서 BBox 번호를 추출합니다.

        음성 명령에서 사용자가 지정한 객체의 바운딩 박스 번호를 추출합니다.
        예: "3번 상품을 담아줘" → 3

        Args:
            text: 자연어 문장 (예: "3번 상품을 담아줘", "첫 번째 것")

        Returns:
            Optional[int]: 추출된 bbox 번호 또는 None (실패 시)

        사용 예:
            bbox = await llm_client.extract_bbox_number("3번 상품을 담아줘")
            if bbox:
                print(f"선택된 bbox: {bbox}")
        """
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
        """
        자연어 문장의 의도와 엔티티를 추출합니다.

        사용자 음성 명령의 의도(intent)와 관련 엔티티를 파악합니다.
        예: "비건 사과 검색해줘" → {"intent": "search", "entities": {"query": "비건 사과"}}

        Args:
            text: 자연어 문장 (예: "비건 사과 검색해줘", "장바구니 확인")

        Returns:
            Optional[Dict[str, Any]]: 의도 및 엔티티 정보 또는 None (실패 시)
                - intent: 사용자 의도 (예: "search", "add_to_cart", "checkout")
                - entities: 추출된 엔티티 정보 (dict)

        사용 예:
            result = await llm_client.detect_intent("비건 사과 검색해줘")
            if result:
                print(f"의도: {result['intent']}")
                print(f"엔티티: {result.get('entities', {})}")
        """
        endpoint = f'{self._base_url}/llm/intent_detection'
        payload = {'text': text}

        logger.debug('LLM intent_detection request to %s with text=%r', endpoint, text)

        data = await self._request_with_retry('GET', endpoint, payload)
        if not data or not data.get('intent'):
            logger.warning('LLM intent response missing intent: %s', data)
            return None
        return data

    def record_fallback(self) -> None:
        """
        LLM 실패 후 기본 로직(fallback)을 사용했음을 기록합니다.

        LLM 서비스가 실패했을 때 대체 로직을 사용한 횟수를 통계에 기록합니다.
        이 정보는 get_stats_snapshot()을 통해 조회할 수 있습니다.

        사용 예:
            sql_query = await llm_client.generate_search_query("비건 사과")
            if not sql_query:
                # 기본 로직 사용
                sql_query = generate_default_query("비건 사과")
                llm_client.record_fallback()
        """
        with self._metrics_lock:
            self._metrics['fallback_count'] += 1

    def get_stats_snapshot(self) -> Dict[str, float]:
        """
        LLM 호출 통계 스냅샷을 반환합니다.

        현재까지 누적된 LLM 호출 통계를 반환합니다.
        대시보드나 모니터링 시스템에서 사용됩니다.

        Returns:
            Dict[str, float]: 통계 정보를 담은 딕셔너리
                - success_rate: 성공률 (%)
                - avg_response_time: 평균 응답 시간 (ms)
                - success_count: 성공 횟수
                - failure_count: 실패 횟수
                - fallback_count: 폴백 사용 횟수

        사용 예:
            stats = llm_client.get_stats_snapshot()
            print(f"LLM 성공률: {stats['success_rate']}%")
            print(f"평균 응답 시간: {stats['avg_response_time']}ms")
        """
        with self._metrics_lock:
            success = self._metrics['success_count']
            failure = self._metrics['failure_count']
            total = success + failure
            success_rate = (success / total * 100.0) if total else 0.0
            avg_latency = (self._metrics['total_latency_ms'] / success) if success else 0.0
            return {
                'success_rate': round(success_rate, 1),
                'avg_response_time': round(avg_latency, 1),
                'failure_count': int(failure),
                'fallback_count': int(self._metrics['fallback_count']),
                'success_count': int(success),
            }

    def _record_success(self, latency_ms: float) -> None:
        """
        성공한 호출과 응답 시간을 기록합니다. (내부 메서드)

        Args:
            latency_ms: 응답 시간 (밀리초)
        """
        with self._metrics_lock:
            self._metrics['success_count'] += 1
            self._metrics['total_latency_ms'] += max(0.0, latency_ms)

    def _record_failure(self) -> None:
        """실패한 호출을 기록합니다. (내부 메서드)"""
        with self._metrics_lock:
            self._metrics['failure_count'] += 1
