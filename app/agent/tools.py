# Agent 工具定义 - 用 @tool 装饰器包装 VehicleService(规格第9节)
# 运行指南:
#   tools = build_vehicle_tools(service)
#   llm.bind_tools(tools)
#   Agent 通过 bind_tools 让 LLM 决定调用哪个工具

from typing import Any

from langchain_core.tools import tool

from app.context.schemas import VehicleState
from app.tools.schemas import ToolResult
from app.tools.vehicle import VehicleService


def build_vehicle_tools(service: VehicleService) -> list[Any]:
    """构造绑定到指定 VehicleService 的工具列表(每次调用生成新实例)。"""

    @tool
    def get_vehicle_status() -> str:
        """查询当前车辆状态,包括车速、电量、车内温度、空调状态、位置、座椅、车窗、后备箱、雨刮、车灯等。"""
        result: ToolResult = service.get_vehicle_status()
        return result.output

    @tool
    def get_cabin_temperature() -> str:
        """查询当前车内温度。"""
        result: ToolResult = service.get_cabin_temperature()
        return result.output

    @tool
    def set_temperature(temperature_c: float) -> str:
        """设置目标车内温度(摄氏度,允许范围 18-30)。"""
        result: ToolResult = service.set_temperature(temperature_c)
        return result.output

    @tool
    def set_ac(enabled: bool) -> str:
        """开启或关闭空调。"""
        result: ToolResult = service.set_ac(enabled)
        return result.output

    @tool
    def set_seat_position(
        seat: str,
        slide_percent: int | None = None,
        backrest_angle_deg: int | None = None,
    ) -> str:
        """设置座椅位置(seat: driver/passenger; slide_percent: 0-100 前后滑动百分比; backrest_angle_deg: 0-180 靠背角度)。至少指定一个参数。"""
        result: ToolResult = service.set_seat_position(seat, slide_percent, backrest_angle_deg)
        return result.output

    @tool
    def set_seat_ventilation(seat: str, level: int) -> str:
        """设置座椅通风档位(seat: driver/passenger; level: 0=关闭, 1-3=档位)。"""
        result: ToolResult = service.set_seat_ventilation(seat, level)
        return result.output

    @tool
    def set_seat_massage(seat: str, level: int) -> str:
        """设置座椅按摩档位(seat: driver/passenger; level: 0=关闭, 1-3=档位)。"""
        result: ToolResult = service.set_seat_massage(seat, level)
        return result.output

    @tool
    def set_seat_heating(seat: str, level: int) -> str:
        """设置座椅加热档位(seat: driver/passenger; level: 0=关闭, 1-3=档位)。"""
        result: ToolResult = service.set_seat_heating(seat, level)
        return result.output

    @tool
    def set_window(window: str, open_percent: int) -> str:
        """设置车窗开启百分比(window: all/driver_front/passenger_front/driver_rear/passenger_rear; open_percent: 0=全关,100=全开)。车速过高时禁止开窗。"""
        result: ToolResult = service.set_window(window, open_percent)
        return result.output

    @tool
    def set_trunk(open: bool) -> str:
        """开启或关闭后备箱(open: true=开启,false=关闭)。车速过高时禁止操作。"""
        result: ToolResult = service.set_trunk(open)
        return result.output

    @tool
    def set_wiper(level: str) -> str:
        """设置雨刮档位(level: off/low/medium/high/auto)。"""
        result: ToolResult = service.set_wiper(level)
        return result.output

    @tool
    def set_light(mode: str) -> str:
        """设置车灯模式(mode: off/parking/low_beam/high_beam/auto)。"""
        result: ToolResult = service.set_light(mode)
        return result.output

    return [
        get_vehicle_status,
        get_cabin_temperature,
        set_temperature,
        set_ac,
        set_seat_position,
        set_seat_ventilation,
        set_seat_massage,
        set_seat_heating,
        set_window,
        set_trunk,
        set_wiper,
        set_light,
    ]


def format_state_for_prompt(state: VehicleState) -> str:
    """格式化车辆状态供 prompt 注入。"""
    wiper_cn = {"off": "关闭", "low": "低档", "medium": "中档", "high": "高档", "auto": "自动"}
    light_cn = {"off": "关闭", "parking": "示宽灯", "low_beam": "近光灯", "high_beam": "远光灯", "auto": "自动"}
    seats = state.seats
    windows = state.windows
    return (
        f"车速 {state.speed_kmh:.0f} km/h,电量 {state.battery_percent:.0f}%,"
        f"车内温度 {state.cabin_temperature_c:.1f}℃,空调{'开启' if state.ac_enabled else '关闭'}。"
        f"座椅通风(驾{seats.driver_ventilation_level}/副{seats.passenger_ventilation_level}),"
        f"按摩(驾{seats.driver_massage_level}/副{seats.passenger_massage_level}),"
        f"加热(驾{seats.driver_heating_level}/副{seats.passenger_heating_level})。"
        f"车窗(前驾{windows.driver_front}%/前副{windows.passenger_front}%/"
        f"后驾{windows.driver_rear}%/后副{windows.passenger_rear}%),"
        f"后备箱{'开' if state.trunk_open else '关'},"
        f"雨刮{wiper_cn.get(state.wiper_level.value, state.wiper_level.value)},"
        f"车灯{light_cn.get(state.light_mode.value, state.light_mode.value)}。"
    )
