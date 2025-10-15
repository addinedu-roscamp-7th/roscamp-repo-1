"""
Redis 기반 RobotState 저장소 구현.
"""
from __future__ import annotations

import inspect
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import redis.asyncio as redis
from redis.asyncio.client import Pipeline
from redis.exceptions import ResponseError, WatchError

from .constants import RobotStatus, RobotType
from .robot_state_backend import RobotState, RobotStateBackend

logger = logging.getLogger(__name__)


def _bool_to_str(value: bool) -> str:
    return "1" if value else "0"


def _str_to_bool(value: Optional[str]) -> bool:
    return value == "1"


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc)


def _parse_robot_type(value: Optional[str]) -> RobotType:
    if not value:
        return RobotType.PICKEE
    normalized = value.strip().lower()
    for enum_value in RobotType:
        if normalized in {enum_value.value, enum_value.name.lower()}:
            return enum_value
    logger.warning(
        "Unknown robot_type '%s' found in Redis; defaulting to %s",
        value,
        RobotType.PICKEE.value,
    )
    return RobotType.PICKEE


class RedisRobotStateBackend(RobotStateBackend):
    """Redis 해시를 이용한 상태 저장 구현."""

    def __init__(
        self,
        url: str,
        key_prefix: str = "shopee:robot_state",
        *,
        client: Optional[redis.Redis] = None,
        socket_timeout: Optional[float] = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=socket_timeout,
        )
        self._closed = False
        self._prefix = key_prefix.rstrip(":")

        self._reserve_script = self._client.register_script(
            """
            local key = KEYS[1]
            local order_id = ARGV[1]
            local timestamp = ARGV[2]
            if redis.call('exists', key) == 0 then
                return 0
            end
            if redis.call('hget', key, 'reserved') == '1' then
                return 0
            end
            redis.call('hmset', key,
                'reserved', '1',
                'active_order_id', order_id,
                'last_update', timestamp)
            return 1
            """
        )

        self._release_script = self._client.register_script(
            """
            local key = KEYS[1]
            local order_id = ARGV[1]
            local timestamp = ARGV[2]
            if redis.call('exists', key) == 0 then
                return 0
            end
            local current_order = redis.call('hget', key, 'active_order_id')
            if order_id ~= '' and current_order ~= order_id then
                return 0
            end
            redis.call('hmset', key,
                'reserved', '0',
                'active_order_id', '',
                'last_update', timestamp)
            return 1
            """
        )

        self._maintenance_script = self._client.register_script(
            """
            local key = KEYS[1]
            local enabled = ARGV[1]
            local timestamp = ARGV[2]
            if redis.call('exists', key) == 0 then
                return 0
            end
            redis.call('hmset', key,
                'maintenance_mode', enabled,
                'last_update', timestamp)
            if enabled == '1' then
                redis.call('hmset', key,
                    'reserved', '0',
                    'active_order_id', '')
            end
            return 1
            """
        )

    def _key(self, robot_id: int) -> str:
        return f"{self._prefix}:{robot_id}"

    async def upsert_state(self, state: RobotState) -> None:
        key = self._key(state.robot_id)
        existing = await self._client.hgetall(key)

        reserved_flag = _str_to_bool(existing.get("reserved")) if existing else state.reserved
        maintenance_flag = _str_to_bool(existing.get("maintenance_mode")) if existing else state.maintenance_mode

        mapping: Dict[str, str] = {
            "robot_id": str(state.robot_id),
            "robot_type": state.robot_type.value,
            "status": state.status,
            "reserved": _bool_to_str(reserved_flag),
            "active_order_id": "" if state.active_order_id is None else str(state.active_order_id),
            "battery_level": "" if state.battery_level is None else str(state.battery_level),
            "last_update": state.last_update.isoformat(),
            "maintenance_mode": _bool_to_str(maintenance_flag),
        }
        await self._client.hset(key, mapping=mapping)

    async def get_state(self, robot_id: int) -> Optional[RobotState]:
        data = await self._client.hgetall(self._key(robot_id))
        if not data:
            return None
        return self._to_state(data)

    async def list_states(self, robot_type: Optional[RobotType] = None) -> List[RobotState]:
        states: List[RobotState] = []
        async for key in self._client.scan_iter(match=f"{self._prefix}:*"):
            data = await self._client.hgetall(key)
            if not data:
                continue
            state = self._to_state(data)
            if robot_type and state.robot_type != robot_type:
                continue
            states.append(state)
        return states

    async def list_available(self, robot_type: RobotType) -> List[RobotState]:
        states = await self.list_states(robot_type)
        return [
            state
            for state in states
            if not state.reserved
            and not state.maintenance_mode
            and state.status == RobotStatus.IDLE.value
        ]

    async def try_reserve(self, robot_id: int, order_id: int) -> bool:
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            result = await self._reserve_script(
                keys=[self._key(robot_id)],
                args=[str(order_id), timestamp],
            )
        except ResponseError as exc:
            if "unknown command `evalsha`" in str(exc).lower():
                result = await self._fallback_try_reserve(robot_id, order_id, timestamp)
            else:  # pragma: no cover - unexpected Redis errors bubble up
                raise
        except Exception:
            raise
        if not result:
            logger.warning(
                "Failed to reserve robot %s for order %s via Redis (key missing or already reserved)",
                robot_id,
                order_id,
            )
        return bool(result)

    async def release(self, robot_id: int, order_id: Optional[int] = None) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            result = await self._release_script(
                keys=[self._key(robot_id)],
                args=[str(order_id) if order_id is not None else "", timestamp],
            )
        except ResponseError as exc:
            if "unknown command `evalsha`" in str(exc).lower():
                result = await self._fallback_release(robot_id, order_id, timestamp)
            else:  # pragma: no cover
                raise
        if not result:
            logger.warning(
                "Failed to release robot %s (order=%s) in Redis",
                robot_id,
                order_id if order_id is not None else "any",
            )

    async def mark_offline(self, robot_id: int) -> None:
        key = self._key(robot_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        await self._client.hset(
            key,
            mapping={
                "status": RobotStatus.OFFLINE.value,
                "reserved": "0",
                "active_order_id": "",
                "last_update": timestamp,
            },
        )

    async def set_maintenance_mode(self, robot_id: int, enabled: bool) -> bool:
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            result = await self._maintenance_script(
                keys=[self._key(robot_id)],
                args=[_bool_to_str(enabled), timestamp],
            )
        except ResponseError as exc:
            if "unknown command `evalsha`" in str(exc).lower():
                result = await self._fallback_maintenance(robot_id, enabled, timestamp)
            else:  # pragma: no cover
                raise
        if not result:
            logger.warning(
                "Failed to set maintenance_mode=%s for robot %s in Redis",
                enabled,
                robot_id,
            )
        return bool(result)

    def _to_state(self, data: Dict[str, str]) -> RobotState:
        robot_type = _parse_robot_type(data.get("robot_type"))
        reserved = _str_to_bool(data.get("reserved"))
        maintenance_mode = _str_to_bool(data.get("maintenance_mode"))
        active_order_id = _optional_int(data.get("active_order_id"))
        battery_level = _optional_float(data.get("battery_level"))
        last_update = _parse_datetime(data.get("last_update"))

        return RobotState(
            robot_id=int(data.get("robot_id", "0")),
            robot_type=robot_type,
            status=data.get("status", RobotStatus.IDLE.value),
            reserved=reserved,
            active_order_id=active_order_id,
            battery_level=battery_level,
            last_update=last_update,
            maintenance_mode=maintenance_mode,
        )

    async def close(self) -> None:
        """Redis 커넥션을 정리합니다."""
        if self._closed or not self._client:
            return
        self._closed = True
        if not self._owns_client:
            return
        close_method = getattr(self._client, "close", None)
        if close_method is not None:
            result = close_method()
            if inspect.isawaitable(result):
                await result
        pool = getattr(self._client, "connection_pool", None)
        if pool is None:
            return
        disconnect = getattr(pool, "disconnect", None)
        if disconnect is None:
            return
        try:
            disconnect(inuse_connections=True)  # redis-py 5.x
        except TypeError:
            disconnect()

    async def aclose(self) -> None:
        """close() 별칭."""
        await self.close()

    async def _fallback_try_reserve(self, robot_id: int, order_id: int, timestamp: str) -> int:
        """Fallback reservation logic for environments without Lua support (e.g. fakeredis)."""
        key = self._key(robot_id)
        async with self._client.pipeline() as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    data = await self._client.hgetall(key)
                    if not data or data.get("reserved") == "1":
                        await pipe.unwatch()
                        return 0
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "reserved": "1",
                            "active_order_id": str(order_id),
                            "last_update": timestamp,
                        },
                    )
                    await pipe.execute()
                    await pipe.unwatch()
                    return 1
                except WatchError:
                    continue
                finally:
                    await _reset_pipeline(pipe)

    async def _fallback_release(
        self,
        robot_id: int,
        order_id: Optional[int],
        timestamp: str,
    ) -> int:
        key = self._key(robot_id)
        async with self._client.pipeline() as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    data = await self._client.hgetall(key)
                    if not data:
                        await pipe.unwatch()
                        return 0
                    current_order = data.get("active_order_id") or ""
                    if order_id is not None and current_order != str(order_id):
                        await pipe.unwatch()
                        return 0
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "reserved": "0",
                            "active_order_id": "",
                            "last_update": timestamp,
                        },
                    )
                    await pipe.execute()
                    await pipe.unwatch()
                    return 1
                except WatchError:
                    continue
                finally:
                    await _reset_pipeline(pipe)

    async def _fallback_maintenance(self, robot_id: int, enabled: bool, timestamp: str) -> int:
        key = self._key(robot_id)
        async with self._client.pipeline() as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    data = await self._client.hgetall(key)
                    if not data:
                        await pipe.unwatch()
                        return 0
                    mapping = {
                        "maintenance_mode": _bool_to_str(enabled),
                        "last_update": timestamp,
                    }
                    if enabled:
                        mapping.update(
                            {
                                "reserved": "0",
                                "active_order_id": "",
                            }
                        )
                    pipe.multi()
                    pipe.hset(key, mapping=mapping)
                    await pipe.execute()
                    await pipe.unwatch()
                    return 1
                except WatchError:
                    continue
                finally:
                    await _reset_pipeline(pipe)


async def _reset_pipeline(pipe: Pipeline) -> None:
    """Reset pipeline safely for reuse in fallback flows."""
    try:
        await pipe.reset()
    except AttributeError:
        # Older redis versions expose sync reset; call directly.
        pipe.reset()  # type: ignore[func-returns-value]
