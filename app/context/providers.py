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


class EnvironmentProvider(Protocol):
    """环境状态提供者接口(委托模式,支持场景池等多种实现)。"""

    def get_state(self) -> EnvironmentState:
        """返回当前环境状态。"""
        ...
