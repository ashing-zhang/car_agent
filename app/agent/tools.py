# Agent 工具定义 - 用 @tool 装饰器包装 VehicleService(规格第9节)
# 运行指南:
#   tools = build_vehicle_tools(service)
#   llm.bind_tools(tools)
#   Agent 通过 bind_tools 让 LLM 决定调用哪个工具

from typing import Any

from langchain.tools import tool

from app.context.schemas import VehicleState
from app.tools.schemas import ToolResult
from app.tools.vehicle import VehicleService


def build_vehicle_tools(service: VehicleService) -> list[Any]:
    """构造绑定到指定 VehicleService 的工具列表(每次调用生成新实例)。"""

    @tool
    def get_vehicle_status() -> str:
        """查询当前车辆状态,包括车速、电量、车内温度、空调状态、位置等。"""
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

    return [get_vehicle_status, get_cabin_temperature, set_temperature, set_ac]


def format_state_for_prompt(state: VehicleState) -> str:
    """格式化车辆状态供 prompt 注入。"""
    return (
        f"车速 {state.speed_kmh:.0f} km/h,电量 {state.battery_percent:.0f}%,"
        f"车内温度 {state.cabin_temperature_c:.1f}℃,空调{'开启' if state.ac_enabled else '关闭'}。"
    )
