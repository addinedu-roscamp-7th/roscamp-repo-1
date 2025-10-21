"""
Pytest configuration helpers for async tests.

별도 플러그인을 설치하지 못하는 환경을 고려하여
간단한 asyncio 실행 훅을 제공한다.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure the local source package is used instead of any pre-installed version.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    """테스트 세션에서 사용할 전용 이벤트 루프를 제공합니다."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def pytest_configure(config: pytest.Config) -> None:
    """커스텀 마커 등록."""
    config.addinivalue_line("markers", "asyncio: mark async test")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Function]) -> None:
    """
    동기 테스트에 잘못 적용된 asyncio 마커를 제거한다.

    pytest 8에서는 동기 함수에 asyncio 마커가 남아 있으면 경고가 발생하므로
    실제로 코루틴 함수가 아닌 경우 마커를 삭제하여 경고를 차단한다.
    """
    for item in items:
        marker = item.get_closest_marker("asyncio")
        if marker and not inspect.iscoroutinefunction(item.obj):
            # 개별 테스트에 부착된 마커 제거
            item.own_markers = [m for m in item.own_markers if m.name != "asyncio"]
            # 모듈/클래스 수준에서 부착된 마커 제거
            item.keywords.pop("asyncio", None)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """
    async 테스트 함수를 asyncio 이벤트 루프로 실행한다.

    pytest-asyncio 등의 외부 플러그인이 없어도 코루틴 함수를 실행할 수 있도록 한다.
    """
    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None  # 기본 처리 경로 사용

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        kwargs: dict[str, Any] = {}
        kwargs.update(pyfuncitem.funcargs)
        loop.run_until_complete(test_func(**kwargs))
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    return True
