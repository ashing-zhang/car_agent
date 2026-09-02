# Environment Tools - 环境上下文工具(规格第9节 environment)
# 运行指南:
#   分层: Agent Tool → EnvironmentService → EnvironmentProvider(委托) → 场景池(app/simulation)
#   get_weather / get_traffic / get_camera_scene 均基于场景池随机选景结果

import logging

from app.context.providers import EnvironmentProvider
from app.context.schemas import EnvironmentState
from app.simulation.providers import SceneEnvironmentProvider
from app.simulation.scene_pool import get_scene_pool
from app.tools.schemas import ToolResult

logger = logging.getLogger(__name__)


class EnvironmentService:
    """环境服务层,委托 EnvironmentProvider 提供结构化环境上下文。"""

    def __init__(self, provider: EnvironmentProvider) -> None:
        """注入环境状态提供者。"""
        self._provider = provider
        self._cached: EnvironmentState | None = None

    def _get_state(self) -> EnvironmentState:
        """获取(并缓存)当前环境状态。"""
        if self._cached is None:
            self._cached = self._provider.get_state()
        return self._cached

    def refresh(self) -> None:
        """清除缓存,下次调用重新随机选景。"""
        self._cached = None

    def get_weather(self) -> ToolResult:
        """查询当前天气。"""
        state = self._get_state()
        return ToolResult(
            success=True,
            tool_name="get_weather",
            output=f"当前天气:{state.weather},置信度 {state.confidence:.0%}。",
            data={"weather": state.weather, "confidence": state.confidence},
        )

    def get_traffic(self) -> ToolResult:
        """查询当前交通密度。"""
        state = self._get_state()
        return ToolResult(
            success=True,
            tool_name="get_traffic",
            output=f"当前交通密度:{state.traffic_density}。",
            data={"traffic_density": state.traffic_density},
        )

    def get_camera_scene(self) -> ToolResult:
        """返回当前场景的完整结构化解析。"""
        self.refresh()
        state = self._get_state()
        parts = [
            f"天气 {state.weather}",
            f"交通 {state.traffic_density}",
            f"道路施工 {'是' if state.road_work else '否'}",
            f"行人 {'有' if state.pedestrians else '无'}",
        ]
        return ToolResult(
            success=True,
            tool_name="get_camera_scene",
            output="; ".join(parts) + f"。置信度 {state.confidence:.0%}。",
            data=state.model_dump(mode="json"),
        )


ENVIRONMENT_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "get_weather",
        "description": "查询当前天气状况。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_traffic",
        "description": "查询当前交通密度。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_camera_scene",
        "description": "返回当前场景的完整结构化解析(天气/交通/施工/行人)。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def execute_environment_tool(name: str, arguments: dict, service: EnvironmentService) -> ToolResult:
    """分派到 EnvironmentService 对应方法。"""
    if name == "get_weather":
        return service.get_weather()
    if name == "get_traffic":
        return service.get_traffic()
    if name == "get_camera_scene":
        return service.get_camera_scene()
    return ToolResult(success=False, tool_name=name, output="", error=f"unknown_tool:{name}")


_default_service: EnvironmentService | None = None


def get_environment_service() -> EnvironmentService:
    """获取默认 EnvironmentService 单例(场景池驱动,随机选景提供环境状态)。"""
    global _default_service
    if _default_service is None:
        _default_service = EnvironmentService(SceneEnvironmentProvider(get_scene_pool()))
    return _default_service
