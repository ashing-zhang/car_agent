# 会话级车辆状态仓储 - VehicleStateStore 与 contextvars 会话上下文
# 运行指南:
#   from app.simulation.vehicle_state_store import (
#       set_session_context, get_session_context, reset_session_context,
#       VehicleStateOverride, build_store, get_vehicle_state_store,
#   )
#   tokens = set_session_context(user_id="u1", session_id="s1")
#   try:
#       store = get_vehicle_state_store()
#       store.save_override("u1", "s1", VehicleStateOverride(target_temperature_c=24))
#       loaded = store.load_override("u1", "s1")
#   finally:
#       reset_session_context(tokens)
#   后端选择: 配置 simulation.state_store_backend 控制 in_memory / redis(未来扩展)

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from functools import lru_cache
from typing import Protocol

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SESSION_USER_ID: ContextVar[str | None] = ContextVar("session_user_id", default=None)
_SESSION_SESSION_ID: ContextVar[str | None] = ContextVar("session_session_id", default=None)


class VehicleStateOverride(BaseModel):
    """会话级车辆状态覆盖(仅在用户显式操作时非 None)。"""

    ac_enabled: bool | None = None
    target_temperature_c: float | None = None
    cabin_temperature_c: float | None = None


class VehicleStateStore(Protocol):
    """车辆状态仓储协议(支持开闭原则,可扩展 Redis 等后端)。"""

    @property
    def backend(self) -> str:
        """返回后端标识字符串。"""
        ...

    def save_override(self, user_id: str, session_id: str, override: VehicleStateOverride) -> None:
        """持久化指定会话的覆盖值。"""
        ...

    def load_override(self, user_id: str, session_id: str) -> VehicleStateOverride:
        """读取指定会话覆盖值,不存在时返回全 None 实例。"""
        ...

    def clear_override(self, user_id: str, session_id: str) -> None:
        """清除指定会话覆盖值(幂等)。"""
        ...


class InMemoryVehicleStateStore:
    """内存后端实现:进程内 dict 存储,适合开发与单元测试。"""

    def __init__(self) -> None:
        """初始化空存储。"""
        self._data: dict[tuple[str, str], VehicleStateOverride] = {}

    @property
    def backend(self) -> str:
        """返回后端标识。"""
        return "in_memory"

    def save_override(self, user_id: str, session_id: str, override: VehicleStateOverride) -> None:
        """保存覆盖值到内存 dict。"""
        self._data[(user_id, session_id)] = override.model_copy(deep=True)
        logger.debug("Override saved for %s/%s: %s", user_id, session_id, override.model_dump())

    def load_override(self, user_id: str, session_id: str) -> VehicleStateOverride:
        """读取覆盖值,缺失时返回空 VehicleStateOverride。"""
        stored = self._data.get((user_id, session_id))
        if stored is None:
            logger.debug("Override not found for %s/%s, returning empty", user_id, session_id)
            return VehicleStateOverride()
        return stored.model_copy(deep=True)

    def clear_override(self, user_id: str, session_id: str) -> None:
        """移除覆盖值,不存在时静默成功。"""
        key = (user_id, session_id)
        if key in self._data:
            del self._data[key]
            logger.debug("Override cleared for %s/%s", user_id, session_id)


def build_store(backend: str) -> VehicleStateStore:
    """根据后端名称构造 VehicleStateStore 实例(配置驱动,开闭原则)。"""
    backend = backend.lower()
    if backend == "in_memory":
        logger.info("Building VehicleStateStore with in_memory backend")
        return InMemoryVehicleStateStore()
    msg = f"Unsupported VehicleStateStore backend: {backend} (supported: in_memory)"
    logger.error(msg)
    raise ValueError(msg)


@lru_cache(maxsize=1)
def get_vehicle_state_store() -> VehicleStateStore:
    """获取默认 VehicleStateStore 单例(配置驱动)。"""
    try:
        from app.config import get_app_config
        backend = get_app_config().simulation.state_store_backend
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to load simulation config, fallback to in_memory: %s", exc)
        backend = "in_memory"
    return build_store(backend)


def set_session_context(user_id: str, session_id: str) -> tuple[Token, Token]:
    """设置当前上下文的 user_id 与 session_id,返回 reset 用的 Token 对。"""
    t1 = _SESSION_USER_ID.set(user_id)
    t2 = _SESSION_SESSION_ID.set(session_id)
    logger.debug("Session context set: user=%s session=%s", user_id, session_id)
    return (t1, t2)


def get_session_context() -> tuple[str | None, str | None]:
    """读取当前 (user_id, session_id),未设置时返回 (None, None)。"""
    return (_SESSION_USER_ID.get(), _SESSION_SESSION_ID.get())


def reset_session_context(tokens: tuple[Token, Token]) -> None:
    """根据 set_session_context 返回的 Token 恢复上下文状态。"""
    t1, t2 = tokens
    _SESSION_USER_ID.reset(t1)
    _SESSION_SESSION_ID.reset(t2)
    logger.debug("Session context reset")
