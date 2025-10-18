"""
Async integration tests for LLMClient using HTTP mocking.
"""
from __future__ import annotations

import respx
import httpx
import pytest

from main_service.llm_client import LLMClient


@pytest.mark.asyncio
async def test_generate_search_query_success() -> None:
    client = LLMClient("http://llm.example")
    with respx.mock(base_url="http://llm.example") as router:
        router.get("/llm/search_query").respond(json={"sql_query": "name LIKE '%사과%'"})
        result = await client.generate_search_query("비건 사과")
    assert result == "name LIKE '%사과%'"


@pytest.mark.asyncio
async def test_generate_search_query_missing_field() -> None:
    client = LLMClient("http://llm.example")
    with respx.mock(base_url="http://llm.example") as router:
        router.get("/llm/search_query").respond(json={})
        result = await client.generate_search_query("사과")
    assert result is None


@pytest.mark.asyncio
async def test_extract_bbox_number_success() -> None:
    client = LLMClient("http://llm.example")
    with respx.mock(base_url="http://llm.example") as router:
        router.get("/llm/bbox").respond(json={"bbox": 2})
        result = await client.extract_bbox_number("2번 집어줘")
    assert result == 2


@pytest.mark.asyncio
async def test_extract_bbox_number_invalid_value() -> None:
    client = LLMClient("http://llm.example")
    with respx.mock(base_url="http://llm.example") as router:
        router.get("/llm/bbox").respond(json={"bbox": "not_an_int"})
        result = await client.extract_bbox_number("invalid")
    assert result is None


@pytest.mark.asyncio
async def test_detect_intent_success() -> None:
    client = LLMClient("http://llm.example")
    with respx.mock(base_url="http://llm.example") as router:
        router.get("/llm/intent_detection").respond(
            json={"intent": "Move_place", "entities": {"place_name": "xx", "action": "move"}}
        )
        result = await client.detect_intent("피키야, xx로 이동해줘")
    assert result == {"intent": "Move_place", "entities": {"place_name": "xx", "action": "move"}}


@pytest.mark.asyncio
async def test_detect_intent_missing_intent() -> None:
    client = LLMClient("http://llm.example")
    with respx.mock(base_url="http://llm.example") as router:
        router.get("/llm/intent_detection").respond(json={})
        result = await client.detect_intent("invalid")
    assert result is None
