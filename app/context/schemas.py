# 上下文数据 Schema - 统一车辆/环境/导航状态定义
# 运行指南: 被各 Adapter 转换为这些 schema,Agent 层只依赖这些类型

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class WiperLevel(StrEnum):
    """雨刮档位枚举。"""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class LightMode(StrEnum):
    """车灯模式枚举。"""

    OFF = "off"
    PARKING = "parking"
    LOW_BEAM = "low_beam"
    HIGH_BEAM = "high_beam"
    AUTO = "auto"


class SeatPosition(BaseModel):
    """座椅位置(前后滑动百分比与靠背角度)。"""

    slide_percent: int = Field(default=50, ge=0, le=100)
    backrest_angle_deg: int = Field(default=90, ge=0, le=180)


class WindowsState(BaseModel):
    """四扇车窗开启百分比(0=全关,100=全开)。"""

    driver_front: int = Field(default=0, ge=0, le=100)
    passenger_front: int = Field(default=0, ge=0, le=100)
    driver_rear: int = Field(default=0, ge=0, le=100)
    passenger_rear: int = Field(default=0, ge=0, le=100)


class SeatsState(BaseModel):
    """座椅状态集合(驾驶位与副驾驶位)。"""

    driver_position: SeatPosition = SeatPosition()
    passenger_position: SeatPosition = SeatPosition()
    driver_ventilation_level: int = Field(default=0, ge=0, le=3)
    passenger_ventilation_level: int = Field(default=0, ge=0, le=3)
    driver_massage_level: int = Field(default=0, ge=0, le=3)
    passenger_massage_level: int = Field(default=0, ge=0, le=3)
    driver_heating_level: int = Field(default=0, ge=0, le=3)
    passenger_heating_level: int = Field(default=0, ge=0, le=3)


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

    seats: SeatsState = SeatsState()
    windows: WindowsState = WindowsState()
    trunk_open: bool = False
    wiper_level: WiperLevel = WiperLevel.OFF
    light_mode: LightMode = LightMode.OFF


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
