# Context Provider 端口协议 - 车辆/环境状态提供者接口(委托模式)
# 运行指南:
#   服务层(app/tools)依赖本端口,基础设施层(app/simulation)提供实现
#   新增状态来源(如真实车机、其他仿真器)实现本协议即可接入,无需修改服务层

from typing import Protocol

from app.context.schemas import EnvironmentState, VehicleState


class VehicleProvider(Protocol):
    """车辆能力提供者接口(委托模式,支持场景池等多种实现)。"""

    def get_state(self) -> VehicleState:
        """返回当前车辆状态。"""
        ...

    def set_cabin_temperature(self, temperature_c: float) -> None:
        """设置目标车内温度。"""
        ...

    def set_ac(self, enabled: bool) -> None:
        """开启或关闭空调。"""
        ...

    def set_seat_position(
        self,
        seat: str,
        slide_percent: int | None = None,
        backrest_angle_deg: int | None = None,
    ) -> None:
        """设置座椅位置(seat: driver/passenger)。"""
        ...

    def set_seat_ventilation(self, seat: str, level: int) -> None:
        """设置座椅通风档位(seat: driver/passenger, level: 0-3)。"""
        ...

    def set_seat_massage(self, seat: str, level: int) -> None:
        """设置座椅按摩档位(seat: driver/passenger, level: 0-3)。"""
        ...

    def set_seat_heating(self, seat: str, level: int) -> None:
        """设置座椅加热档位(seat: driver/passenger, level: 0-3)。"""
        ...

    def set_window(self, window: str, open_percent: int) -> None:
        """设置车窗开启百分比(window: driver_front/passenger_front/driver_rear/passenger_rear/all)。"""
        ...

    def set_trunk(self, open: bool) -> None:
        """开启或关闭后备箱。"""
        ...

    def set_wiper(self, level: str) -> None:
        """设置雨刮档位(off/low/medium/high/auto)。"""
        ...

    def set_light(self, mode: str) -> None:
        """设置车灯模式(off/parking/low_beam/high_beam/auto)。"""
        ...


class EnvironmentProvider(Protocol):
    """环境状态提供者接口(委托模式,支持场景池等多种实现)。"""

    def get_state(self) -> EnvironmentState:
        """返回当前环境状态。"""
        ...
