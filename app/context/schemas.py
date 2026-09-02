# 上下文数据 Schema - 统一车辆/环境/导航状态定义
# 运行指南: 被各 Adapter 转换为这些 schema,Agent 层只依赖这些类型

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class VehicleState(BaseModel):
    """车辆状态(规格第8节),场景池 Provider 负责生成为此 schema。"""

    speed_kmh: float = Field(ge=0)
    battery_percent: float = Field(ge=0, le=100)
    cabin_temperature_c: float
    ac_enabled: bool
    target_temperature_c: float
    latitude: float
    longitude: float
    current_road: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EnvironmentState(BaseModel):
    """环境状态(来自场景池随机选景),规格第17节。"""

    weather: str = "clear"
    traffic_density: str = "low"
    road_work: bool = False
    pedestrians: bool = False
    confidence: float = Field(default=0.8, ge=0, le=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NavigationState(BaseModel):
    """导航状态。"""

    active: bool = False
    destination: str | None = None
    estimated_time_minutes: int | None = None
    remaining_distance_km: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
