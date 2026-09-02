# 场景池驱动 Provider - VehicleProvider / EnvironmentProvider 的场景池实现
# 运行指南:
#   provider = SceneVehicleProvider(get_scene_pool())
#   state = provider.get_state()
#   温度/空调控制会作为运行时覆盖,后续 get_state 保持生效(叠加在随机场景之上)

import logging
from datetime import datetime, timezone

from app.context.schemas import EnvironmentState, VehicleState
from app.simulation.scene_pool import ScenePool

logger = logging.getLogger(__name__)


class SceneVehicleProvider:
    """基于场景池的车辆状态提供者:随机选景 + 运行时控制覆盖。"""

    def __init__(self, pool: ScenePool) -> None:
        """注入场景池,初始化运行时控制覆盖为未设置。"""
        self._pool = pool
        self._cabin_temperature_c: float | None = None
        self._target_temperature_c: float | None = None
        self._ac_enabled: bool | None = None

    def get_state(self) -> VehicleState:
        """随机选取场景并叠加运行时控制覆盖,返回车辆状态。"""
        scene = self._pool.pick()
        vehicle = scene.vehicle
        state = VehicleState(
            speed_kmh=vehicle.speed_kmh,
            battery_percent=vehicle.battery_percent,
            cabin_temperature_c=(
                vehicle.cabin_temperature_c
                if self._cabin_temperature_c is None
                else self._cabin_temperature_c
            ),
            ac_enabled=vehicle.ac_enabled if self._ac_enabled is None else self._ac_enabled,
            target_temperature_c=(
                vehicle.target_temperature_c
                if self._target_temperature_c is None
                else self._target_temperature_c
            ),
            latitude=vehicle.latitude,
            longitude=vehicle.longitude,
            current_road=vehicle.current_road,
            timestamp=datetime.now(timezone.utc),
        )
        logger.debug("Vehicle state from scene %s: %.0f km/h", scene.name, state.speed_kmh)
        return state

    def set_cabin_temperature(self, temperature_c: float) -> None:
        """设置目标车内温度并开启空调(运行时覆盖场景值)。"""
        self._target_temperature_c = temperature_c
        self._cabin_temperature_c = temperature_c
        if not self._ac_enabled:
            self._ac_enabled = True
        logger.info("Scene vehicle cabin temperature overridden to %.1f°C", temperature_c)

    def set_ac(self, enabled: bool) -> None:
        """切换空调开关(运行时覆盖场景值)。"""
        self._ac_enabled = enabled
        logger.info("Scene vehicle AC %s", "enabled" if enabled else "disabled")


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
