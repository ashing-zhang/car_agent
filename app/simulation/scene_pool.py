# 场景池 - 替代 CARLA 仿真的环境变量来源(随机选取)
# 运行指南:
#   pool = get_scene_pool()   # 从 configs/scene_pool.yaml 加载的单例
#   scene = pool.pick()       # 随机选取一个场景(SceneDefinition)
#   自定义场景: 在 configs/scene_pool.yaml 的 scenes 列表中追加条目即可

import logging
import random
from functools import lru_cache

from pydantic import BaseModel, Field

from app.config import load_yaml_config

logger = logging.getLogger(__name__)


class SceneVehicleSpec(BaseModel):
    """场景车辆状态规格(对应 configs/scene_pool.yaml 的 vehicle 字段)。"""

    speed_kmh: float = Field(default=0.0, ge=0)
    battery_percent: float = Field(default=80.0, ge=0, le=100)
    cabin_temperature_c: float = 22.0
    ac_enabled: bool = False
    target_temperature_c: float = 22.0
    latitude: float = 40.0123
    longitude: float = 116.4567
    current_road: str | None = None


class SceneEnvironmentSpec(BaseModel):
    """场景环境状态规格(对应 configs/scene_pool.yaml 的 environment 字段)。"""

    weather: str = "clear"
    traffic_density: str = "low"
    road_work: bool = False
    pedestrians: bool = False
    confidence: float = Field(default=0.85, ge=0, le=1)


class SceneDefinition(BaseModel):
    """场景池中的单个场景定义。"""

    name: str
    vehicle: SceneVehicleSpec = SceneVehicleSpec()
    environment: SceneEnvironmentSpec = SceneEnvironmentSpec()


class ScenePool:
    """场景池:从配置加载场景集合,支持随机选取。"""

    def __init__(self, scenes: list[SceneDefinition], rng: random.Random | None = None) -> None:
        """注入场景列表与可选随机源,场景池不允许为空。"""
        if not scenes:
            raise ValueError("ScenePool requires at least one scene")
        self._scenes = scenes
        self._rng = rng or random.Random()

    @classmethod
    def from_config(cls, filename: str = "scene_pool.yaml") -> "ScenePool":
        """从 configs 目录加载场景池配置。"""
        data = load_yaml_config(filename)
        raw = data.get("scenes") or []
        scenes = [SceneDefinition(**item) for item in raw]
        logger.info("ScenePool loaded %d scenes from %s", len(scenes), filename)
        return cls(scenes)

    def pick(self) -> SceneDefinition:
        """随机选取一个场景。"""
        scene = self._rng.choice(self._scenes)
        logger.debug("Scene picked: %s", scene.name)
        return scene

    @property
    def size(self) -> int:
        """返回场景池大小。"""
        return len(self._scenes)


@lru_cache(maxsize=1)
def get_scene_pool() -> ScenePool:
    """获取场景池单例(配置驱动)。"""
    return ScenePool.from_config()
