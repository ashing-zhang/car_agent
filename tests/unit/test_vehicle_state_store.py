# VehicleStateStore / 会话级车辆状态仓储 单元测试
# 运行指南:
#   pytest tests/unit/test_vehicle_state_store.py -v
#   覆盖: InMemory 后端 CRUD、会话上下文、SceneVehicleProvider 会话隔离、偏好初始化与弱偏好写入

from __future__ import annotations

from app.config import PolicyConfig
from app.memory.models import Memory, MemoryType
from app.simulation.providers import SceneVehicleProvider
from app.simulation.scene_pool import SceneDefinition, ScenePool, SceneVehicleSpec
from app.simulation.vehicle_state_store import (
    InMemoryVehicleStateStore,
    VehicleStateOverride,
    build_store,
    get_session_context,
    reset_session_context,
    set_session_context,
)
from app.tools.vehicle import VehicleService


def _make_pool() -> ScenePool:
    """构造只含一个固定场景的 ScenePool,消除随机性便于断言。"""
    scene = SceneDefinition(
        name="fixed",
        vehicle=SceneVehicleSpec(
            speed_kmh=50,
            battery_percent=60,
            cabin_temperature_c=22,
            ac_enabled=False,
            target_temperature_c=22,
        ),
    )
    return ScenePool([scene])


class FakeMemoryService:
    """伪 MemoryService,预置偏好并记录 remember() 调用。"""

    def __init__(self, preferences: dict[str, Memory] | None = None) -> None:
        self._prefs: dict[str, Memory] = preferences or {}
        self.remember_calls: list[tuple[str, str, str]] = []

    def recall_preference(self, user_id: str, predicate: str):
        """返回预置偏好或 None。"""
        key = f"{user_id}:{predicate}"
        return self._prefs.get(key)

    def remember(self, user_message: str, user_id: str, session_id: str):
        """记录调用,返回空列表(不影响其他逻辑)。"""
        self.remember_calls.append((user_message, user_id, session_id))
        return []


# ---------------- VehicleStateOverride / InMemory 后端 ----------------


def test_override_default_all_none() -> None:
    """新构造的 VehicleStateOverride 三个字段均为 None。"""
    ov = VehicleStateOverride()
    assert ov.ac_enabled is None
    assert ov.target_temperature_c is None
    assert ov.cabin_temperature_c is None


def test_in_memory_save_and_load_roundtrip() -> None:
    """保存覆盖值后再读取,字段完全一致。"""
    store = InMemoryVehicleStateStore()
    ov = VehicleStateOverride(ac_enabled=True, target_temperature_c=24, cabin_temperature_c=24)
    store.save_override("u1", "s1", ov)
    loaded = store.load_override("u1", "s1")
    assert loaded.ac_enabled is True
    assert loaded.target_temperature_c == 24
    assert loaded.cabin_temperature_c == 24


def test_in_memory_load_missing_returns_empty() -> None:
    """不存在的用户/会话读取返回全 None,不自动持久化空对象。"""
    store = InMemoryVehicleStateStore()
    loaded = store.load_override("no_user", "no_session")
    assert loaded.ac_enabled is None
    assert loaded.target_temperature_c is None
    assert loaded.cabin_temperature_c is None


def test_in_memory_clear_override_idempotent() -> None:
    """清除存在的会话覆盖后读回为全 None;清除不存在的会话不抛异常。"""
    store = InMemoryVehicleStateStore()
    store.save_override("u1", "s1", VehicleStateOverride(ac_enabled=True))
    store.clear_override("u1", "s1")
    store.clear_override("u1", "s1")
    store.clear_override("nope", "nope")
    assert store.load_override("u1", "s1").ac_enabled is None


def test_in_memory_user_isolation() -> None:
    """不同用户/会话互不影响。"""
    store = InMemoryVehicleStateStore()
    store.save_override("u1", "s1", VehicleStateOverride(target_temperature_c=24))
    store.save_override("u1", "s2", VehicleStateOverride(target_temperature_c=20))
    store.save_override("u2", "s1", VehicleStateOverride(target_temperature_c=26))
    assert store.load_override("u1", "s1").target_temperature_c == 24
    assert store.load_override("u1", "s2").target_temperature_c == 20
    assert store.load_override("u2", "s1").target_temperature_c == 26


def test_build_store_in_memory_by_default() -> None:
    """无配置时 build_store 返回内存后端。"""
    store = build_store("in_memory")
    assert store.backend == "in_memory"


# ---------------- 会话 contextvars ----------------


def test_session_context_set_and_get() -> None:
    """设置会话上下文后 get_session_context 返回匹配值,reset 后恢复。"""
    before = get_session_context()
    assert before == (None, None)
    tokens = set_session_context("u_test", "s_test")
    try:
        assert get_session_context() == ("u_test", "s_test")
    finally:
        reset_session_context(tokens)
    assert get_session_context() == before


# ---------------- SceneVehicleProvider 会话隔离 ----------------


def test_provider_without_context_uses_global_and_persists() -> None:
    """未设置会话上下文时,provider 回退到 __global__ 占位符,向后兼容。"""
    store = InMemoryVehicleStateStore()
    provider = SceneVehicleProvider(_make_pool(), store=store)
    provider.set_cabin_temperature(25)
    state = provider.get_state()
    assert state.cabin_temperature_c == 25
    assert state.ac_enabled is True
    loaded = store.load_override("__global__", "__global__")
    assert loaded.target_temperature_c == 25


def test_provider_session_isolation() -> None:
    """两个用户同一 provider 各自设置温度,互不干扰。"""
    store = InMemoryVehicleStateStore()
    provider = SceneVehicleProvider(_make_pool(), store=store, apply_preference_on_start=False)

    t1 = set_session_context("alice", "s1")
    try:
        provider.set_cabin_temperature(22)
    finally:
        reset_session_context(t1)

    t2 = set_session_context("bob", "s1")
    try:
        provider.set_cabin_temperature(28)
        bob_state = provider.get_state()
    finally:
        reset_session_context(t2)

    t1 = set_session_context("alice", "s1")
    try:
        alice_state = provider.get_state()
    finally:
        reset_session_context(t1)

    assert alice_state.cabin_temperature_c == 22
    assert bob_state.cabin_temperature_c == 28


def test_provider_set_ac_persists_in_store() -> None:
    """set_ac 保存到仓储,get_state 读出一致状态。"""
    store = InMemoryVehicleStateStore()
    provider = SceneVehicleProvider(_make_pool(), store=store, apply_preference_on_start=False)
    tokens = set_session_context("u", "s")
    try:
        provider.set_ac(True)
        state = provider.get_state()
        assert state.ac_enabled is True
        assert store.load_override("u", "s").ac_enabled is True
        provider.set_ac(False)
        state2 = provider.get_state()
        assert state2.ac_enabled is False
    finally:
        reset_session_context(tokens)


# ---------------- 偏好初始化(用户记忆 -> 会话覆盖) ----------------


def test_provider_applies_preference_on_session_start() -> None:
    """会话首次 get_state 前,从 MemoryService 拷贝偏好温度到覆盖层。"""
    store = InMemoryVehicleStateStore()
    mem = FakeMemoryService(
        preferences={
            "alice:preferred_temperature": Memory(
                id="m1",
                user_id="alice",
                session_id="old",
                type=MemoryType.PREFERENCE,
                predicate="preferred_temperature",
                object="24celsius",
                value=24,
                unit="celsius",
                confidence=0.9,
            )
        }
    )
    provider = SceneVehicleProvider(
        _make_pool(),
        store=store,
        memory_service=mem,
        apply_preference_on_start=True,
        preference_min_confidence=0.6,
    )
    tokens = set_session_context("alice", "fresh_session")
    try:
        state = provider.get_state()
    finally:
        reset_session_context(tokens)
    assert state.cabin_temperature_c == 24
    assert store.load_override("alice", "fresh_session").target_temperature_c == 24


def test_provider_skips_preference_below_min_confidence() -> None:
    """置信度低于阈值的偏好不会被应用。"""
    mem = FakeMemoryService(
        preferences={
            "u:preferred_temperature": Memory(
                id="m1", user_id="u", session_id="old",
                type=MemoryType.PREFERENCE, predicate="preferred_temperature",
                object="25celsius", value=25, confidence=0.3,
            )
        }
    )
    provider = SceneVehicleProvider(
        _make_pool(), memory_service=mem,
        apply_preference_on_start=True, preference_min_confidence=0.6,
    )
    tokens = set_session_context("u", "s")
    try:
        state = provider.get_state()
    finally:
        reset_session_context(tokens)
    assert state.cabin_temperature_c == 22  # 场景默认值


# ---------------- VehicleService 弱偏好写入 ----------------


def test_service_set_temperature_auto_saves_preference() -> None:
    """在会话上下文下调用 set_temperature,会触发 MemoryService.remember。"""
    mem = FakeMemoryService()
    store = InMemoryVehicleStateStore()
    provider = SceneVehicleProvider(_make_pool(), store=store, apply_preference_on_start=False)
    svc = VehicleService(provider, PolicyConfig(), memory_service=mem, auto_save_preference_confidence=0.55)
    tokens = set_session_context("user_x", "session_x")
    try:
        result = svc.set_temperature(23)
    finally:
        reset_session_context(tokens)
    assert result.success is True
    assert len(mem.remember_calls) == 1
    msg, uid, sid = mem.remember_calls[0]
    assert "23" in msg
    assert uid == "user_x"
    assert sid == "session_x"


def test_service_without_context_does_not_save_preference() -> None:
    """未设置会话上下文时,set_temperature 不写入偏好。"""
    mem = FakeMemoryService()
    provider = SceneVehicleProvider(_make_pool(), apply_preference_on_start=False)
    svc = VehicleService(provider, PolicyConfig(), memory_service=mem)
    result = svc.set_temperature(24)
    assert result.success is True
    assert mem.remember_calls == []
