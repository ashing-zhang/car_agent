# Vehicle Tools - 车辆状态查询与控制工具(规格第9节)
# 运行指南:
#   分层: Agent Tool → VehicleService → VehicleProvider(委托) → 场景池(app/simulation)
#   所有工具经 Schema Validation → Policy Validation → Execution → Audit Log(规格第10节)
#   工厂: from app.tools.vehicle import get_vehicle_service

import logging

from app.config import PolicyConfig, get_app_config
from app.context.providers import VehicleProvider
from app.simulation.providers import SceneVehicleProvider
from app.simulation.scene_pool import get_scene_pool
from app.tools.schemas import ToolResult

logger = logging.getLogger(__name__)


class VehicleService:
    """车辆服务层,负责 policy 校验后委托给 VehicleProvider 执行。"""

    def __init__(self, provider: VehicleProvider, policy: PolicyConfig) -> None:
        """注入车辆状态提供者与安全策略配置。"""
        self._provider = provider
        self._policy = policy

    def get_vehicle_status(self) -> ToolResult:
        """查询当前车辆状态。"""
        try:
            state = self._provider.get_state()
            output = (
                f"当前车速 {state.speed_kmh:.0f} km/h,电量 {state.battery_percent:.0f}%,"
                f"车内温度 {state.cabin_temperature_c:.1f}℃。"
            )
            logger.info("get_vehicle_status: %s", output)
            return ToolResult(
                success=True,
                tool_name="get_vehicle_status",
                output=output,
                data=state.model_dump(mode="json"),
            )
        except Exception as exc:
            logger.exception("get_vehicle_status failed")
            return ToolResult(success=False, tool_name="get_vehicle_status", output="", error=str(exc))

    def get_cabin_temperature(self) -> ToolResult:
        """查询当前车内温度。"""
        try:
            state = self._provider.get_state()
            output = f"当前车内温度 {state.cabin_temperature_c:.1f}℃。"
            return ToolResult(
                success=True,
                tool_name="get_cabin_temperature",
                output=output,
                data={"cabin_temperature_c": state.cabin_temperature_c},
            )
        except Exception as exc:
            logger.exception("get_cabin_temperature failed")
            return ToolResult(success=False, tool_name="get_cabin_temperature", output="", error=str(exc))

    def set_temperature(self, temperature_c: float) -> ToolResult:
        """设置目标车内温度,执行前进行 policy 校验(18-30℃)。"""
        if not (self._policy.temperature_min <= temperature_c <= self._policy.temperature_max):
            msg = (
                f"温度 {temperature_c}°C 超出安全范围 "
                f"[{self._policy.temperature_min}, {self._policy.temperature_max}]。"
            )
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_temperature", output=msg, error="policy_violation")
        try:
            self._provider.set_cabin_temperature(temperature_c)
            output = f"已将目标车内温度设置为 {temperature_c}°C。"
            logger.info("set_temperature(%.1f) -> %s", temperature_c, output)
            return ToolResult(
                success=True,
                tool_name="set_temperature",
                output=output,
                data={"target_temperature_c": temperature_c},
            )
        except Exception as exc:
            logger.exception("set_temperature failed")
            return ToolResult(success=False, tool_name="set_temperature", output="", error=str(exc))

    def set_ac(self, enabled: bool) -> ToolResult:
        """开启或关闭空调。"""
        try:
            self._provider.set_ac(enabled)
            output = f"空调已{'开启' if enabled else '关闭'}。"
            logger.info("set_ac(%s) -> %s", enabled, output)
            return ToolResult(
                success=True,
                tool_name="set_ac",
                output=output,
                data={"ac_enabled": enabled},
            )
        except Exception as exc:
            logger.exception("set_ac failed")
            return ToolResult(success=False, tool_name="set_ac", output="", error=str(exc))


VEHICLE_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "get_vehicle_status",
        "description": "查询当前车辆状态,包括车速、电量、车内温度、空调状态、位置等。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_cabin_temperature",
        "description": "查询当前车内温度。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_temperature",
        "description": "设置目标车内温度(摄氏度,范围 18-30)。",
        "input_schema": {
            "type": "object",
            "properties": {"temperature_c": {"type": "number"}},
            "required": ["temperature_c"],
        },
    },
    {
        "name": "set_ac",
        "description": "开启或关闭空调。",
        "input_schema": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    },
]


def execute_vehicle_tool(name: str, arguments: dict, service: VehicleService) -> ToolResult:
    """根据工具名分派到 VehicleService 对应方法。"""
    if name == "get_vehicle_status":
        return service.get_vehicle_status()
    if name == "get_cabin_temperature":
        return service.get_cabin_temperature()
    if name == "set_temperature":
        return service.set_temperature(float(arguments.get("temperature_c", 0)))
    if name == "set_ac":
        return service.set_ac(bool(arguments.get("enabled", False)))
    return ToolResult(success=False, tool_name=name, output="", error=f"unknown_tool:{name}")


_default_service: VehicleService | None = None


def get_vehicle_service() -> VehicleService:
    """获取默认 VehicleService 单例(场景池驱动,随机选景提供车辆状态)。"""
    global _default_service
    if _default_service is not None:
        return _default_service
    policy = get_app_config().policy
    provider = SceneVehicleProvider(get_scene_pool())
    _default_service = VehicleService(provider, policy)
    return _default_service
