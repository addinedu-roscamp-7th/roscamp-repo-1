from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """HTTP client wrapper for LLM endpoints."""

    def __init__(self, base_url: str, timeout: float = 1.5) -> None:
        self._base_url = base_url
        self._timeout = timeout

    async def generate_search_query(self, text: str) -> Optional[str]:
        logger.debug("LLM search_query request text=%s", text)
        # TODO: integrate httpx client
        return None

    async def detect_intent(self, text: str) -> Dict[str, Any]:
        logger.debug("LLM intent_detection request text=%s", text)
        return {}
