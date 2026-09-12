# 场景池驱动 Provider - VehicleProvider / EnvironmentProvider 的场景池实现
# 运行指南:
#   from app.simulation.vehicle_state_store import set_session_context
#   provider = SceneVehicleProvider(get_scene_pool())
#   tokens = set_session_context(user_id="u1", session_id="s1")
#   state = provider.get_state()
#   provider.set_cabin_temperature(24)  # 持久化到会话级 VehicleStateStore
#   偏好记忆(MemoryService)在会话首次加载时自动叠加至覆盖层
#   温度/空调控制保存在 VehicleStateStore 中,进程重启后可通过 redis 后端保留

import logging
from datetime import datetime, timezone

from app.context.schemas import EnvironmentState, VehicleState
from app.simulation.scene_pool import ScenePool


logger = logging.getLogger(__name__)

_GLOBAL_USER = "__global__"
_GLOBAL_SESSION = "__global__"


class SceneVehicleProvider:
    """基于场景池的车辆状态提供者:随机选景 + 会话级状态覆盖 + 用户偏好初始化。

    状态优先级(从高到低):
      1. 用户本次会话中显式设置的覆盖 (VehicleStateStore, 会话级)
      2. 用户持久化偏好记忆 (MemoryService, 跨会话) - 仅会话首次加载时应用一次
      3. 场景池默认值 (随机选景)

    委托模式:本类不直接操作存储,委托 VehicleStateStore/外部 MemoryService 协作。
    """

    def __init__(
        self,
        pool: ScenePool,
        *,
        store: VehicleStateStore | None = None,
        memory_service: object | None = None,
        apply_preference_on_start: bool | None = None,
        preference_min_confidence: float | None = None,
    ) -> None:
        """注入场景池与可选的状态仓储/记忆服务(依赖注入,便于测试)。"""
        self._pool = pool
        self._store: VehicleStateStore = store or get_vehicle_state_store()
        self._memory_service = memory_service
        self._sessions_initialized: set[tuple[str, str]] = set()
        if apply_preference_on_start is None:
            try:
                from app.config import get_app_config
                apply_preference_on_start = get_app_config().simulation.apply_preference_on_session_start
            except Exception:  # noqa: BLE001
                apply_preference_on_start = True
        if preference_min_confidence is None:
            try:
                from app.config import get_app_config
                preference_min_confidence = get_app_config().simulation.preference_based_ac_min_confidence
            except Exception:  # noqa: BLE001
                preference_min_confidence = 0.6
        self._apply_preference_on_start: bool = apply_preference_on_start
        self._preference_min_confidence: float = preference_min_confidence

    # ---------------- 内部工具 ----------------

    def _resolve_context(self) -> tuple[str, str]:
        """解析当前 (user_id, session_id),未设置则回退到全局占位符。"""
        uid, sid = get_session_context()
        return (uid or _GLOBAL_USER, sid or _GLOBAL_SESSION)

    def _ensure_session_initialized(self, user_id: str, session_id: str) -> None:
        """会话首次访问时,将用户偏好记忆写入会话覆盖层(仅一次)。"""
        if (user_id, session_id) in self._sessions_initialized:
            return
        self._sessions_initialized.add((user_id, session_id))
        if not self._apply_preference_on_start:
            return
        if user_id == _GLOBAL_USER:
            return
        pref_temp, pref_ac = self._load_preferences(user_id)
        if pref_temp is None and pref_ac is None:
            return
        existing = self._store.load_override(user_id, session_id)
        merged = VehicleStateOverride(
            ac_enabled=existing.ac_enabled if existing.ac_enabled is not None else pref_ac,
            target_temperature_c=(
                existing.target_temperature_c
                if existing.target_temperature_c is not None
                else pref_temp
            ),
            cabin_temperature_c=(
                existing.cabin_temperature_c
                if existing.cabin_temperature_c is not None
                else pref_temp
            ),
        )
        self._store.save_override(user_id, session_id, merged)
        logger.info(
            "Session %s/%s initialized from memory preferences: temp=%s ac=%s",
            user_id, session_id, pref_temp, pref_ac,
        )

    def _load_preferences(self, user_id: str) -> tuple[float | None, bool | None]:
        """从 MemoryService 读取用户的温度偏好与空调模式偏好(失败安全)。"""
        service = self._memory_service
        if service is None:
            try:
                from app.memory.service import get_memory_service
                service = get_memory_service()
            except Exception as exc:  # noqa: BLE001
                logger.debug("MemoryService unavailable for preferences: %s", exc)
                return None, None
        pref_temp: float | None = None
        pref_ac: bool | None = None
        try:
            mem = service.recall_preference(user_id, "preferred_temperature")
            if mem is not None and mem.confidence >= self._preference_min_confidence:
                pref_temp = float(mem.value) if isinstance(mem.value, (int, float, str)) else None
        except Exception:  # noqa: BLE001
            logger.exception("Failed to recall preferred_temperature for user=%s", user_id)
        try:
            mem = service.recall_preference(user_id, "ac_mode")
            if mem is not None and mem.confidence >= self._preference_min_confidence:
                val = str(mem.value).lower()
                if val in {"cool", "heat", "auto", "dry"}:
                    pref_ac = True
                elif val == "fan":
                    pref_ac = True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to recall ac_mode for user=%s", user_id)
        return pref_temp, pref_ac

    # ---------------- Provider API ----------------

    def get_state(self) -> VehicleState:
        """随机选取场景并按优先级叠加用户覆盖+偏好,返回车辆状态。"""
        user_id, session_id = self._resolve_context()
        self._ensure_session_initialized(user_id, session_id)
        scene = self._pool.pick()
        vehicle = scene.vehicle
        override = self._store.load_override(user_id, session_id)

        def _pick(scene_val: object, override_val: object) -> object:
            return override_val if override_val is not None else scene_val

        state = VehicleState(
            speed_kmh=vehicle.speed_kmh,
            battery_percent=vehicle.battery_percent,
            cabin_temperature_c=float(_pick(vehicle.cabin_temperature_c, override.cabin_temperature_c)),
            ac_enabled=bool(_pick(vehicle.ac_enabled, override.ac_enabled)),
            target_temperature_c=float(_pick(vehicle.target_temperature_c, override.target_temperature_c)),
            latitude=vehicle.latitude,
            longitude=vehicle.longitude,
            current_road=vehicle.current_road,
            timestamp=datetime.now(timezone.utc),
        )
        logger.debug(
            "Vehicle state user=%s session=%s scene=%s: %.0f km/h, AC=%s, T=%.1f°C",
            user_id, session_id, scene.name, state.speed_kmh, state.ac_enabled, state.cabin_temperature_c,
        )
        return state

    def set_cabin_temperature(self, temperature_c: float) -> None:
        """设置目标车内温度并开启空调(保存到会话级状态仓储)。"""
        user_id, session_id = self._resolve_context()
        self._ensure_session_initialized(user_id, session_id)
        override = self._store.load_override(user_id, session_id)
        override.target_temperature_c = temperature_c
        override.cabin_temperature_c = temperature_c
        if override.ac_enabled is not True:
            override.ac_enabled = True
        self._store.save_override(user_id, session_id, override)
        logger.info(
            "Scene vehicle cabin temperature overridden to %.1f°C (user=%s session=%s backend=%s)",
            temperature_c, user_id, session_id, self._store.backend,
        )

    def set_ac(self, enabled: bool) -> None:
        """切换空调开关(保存到会话级状态仓储)。"""
        user_id, session_id = self._resolve_context()
        self._ensure_session_initialized(user_id, session_id)
        override = self._store.load_override(user_id, session_id)
        override.ac_enabled = enabled
        self._store.save_override(user_id, session_id, override)
        logger.info(
            "Scene vehicle AC %s (user=%s session=%s backend=%s)",
            "enabled" if enabled else "disabled", user_id, session_id, self._store.backend,
        )


class SceneEnvironmentProvider:
    """基于场景池的环境状态提供者:随机选景。"""

    def __init__(self, pool: ScenePool) -> None:
        """注入场景池。"""
        self._pool = pool

    def get_state(self) -> EnvironmentState:
        """随机选取场景,返回环境状态。"""
        scene = self._pool.pick()
        state = EnvironmentState(**scene.environment.model_dump())
        logger.info("Environment scene picked: %s (weather=%s, traffic=%s)", scene.name, state.weather, state.traffic_density)
        return state
