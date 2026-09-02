# Navigation Tools - 导航工具(规格第9节 navigation)
# 运行指南:
#   分层: Agent Tool → NavigationService → NavigationProvider(委托) → 地图 API
#   默认 SimulatedNavigationProvider,不依赖外部地图服务(Rule 5)

import logging
from typing import Protocol

from app.context.schemas import NavigationState
from app.tools.schemas import ToolResult

logger = logging.getLogger(__name__)


class NavigationProvider(Protocol):
    """导航能力提供者接口(委托模式)。"""

    def get_status(self) -> NavigationState:
        """返回当前导航状态。"""
        ...

    def search(self, query: str) -> list[dict]:
        """根据关键词搜索目的地。"""
        ...

    def start(self, destination: str) -> bool:
        """开始导航到指定目的地。"""
        ...

    def cancel(self) -> None:
        """取消当前导航。"""
        ...


class SimulatedNavigationProvider:
    """模拟导航提供者,内存维护目的地与路线。"""

    def __init__(self) -> None:
        """初始化无活动导航状态。"""
        self._active: bool = False
        self._destination: str | None = None
        self._eta_minutes: int | None = None
        self._remaining_km: float | None = None

    def get_status(self) -> NavigationState:
        """返回当前导航状态。"""
        return NavigationState(
            active=self._active,
            destination=self._destination,
            estimated_time_minutes=self._eta_minutes,
            remaining_distance_km=self._remaining_km,
        )

    def search(self, query: str) -> list[dict]:
        """根据查询返回模拟目的地列表。"""
        name = query.strip() or "目的地"
        results = [
            {"name": f"{name} 大厦", "address": f"{name}路 1 号", "distance_km": 5.2},
            {"name": f"{name} 广场", "address": f"{name}街 88 号", "distance_km": 8.7},
        ]
        logger.info("Navigation search '%s' -> %d results", query, len(results))
        return results

    def start(self, destination: str) -> bool:
        """开始导航到目的地。"""
        self._active = True
        self._destination = destination
        self._eta_minutes = 18
        self._remaining_km = 5.2
        logger.info("Navigation started to %s", destination)
        return True

    def cancel(self) -> None:
        """取消导航。"""
        self._active = False
        self._destination = None
        self._eta_minutes = None
        self._remaining_km = None
        logger.info("Navigation cancelled")


class NavigationService:
    """导航服务层,policy 校验后委托给 NavigationProvider。"""

    def __init__(self, provider: NavigationProvider) -> None:
        """注入导航提供者。"""
        self._provider = provider

    def get_navigation_status(self) -> ToolResult:
        """查询当前导航状态。"""
        state = self._provider.get_status()
        if not state.active:
            output = "当前没有进行中的导航。"
        else:
            output = (
                f"正在导航到 {state.destination},预计剩余 "
                f"{state.remaining_distance_km} km / {state.estimated_time_minutes} 分钟。"
            )
        return ToolResult(
            success=True,
            tool_name="get_navigation_status",
            output=output,
            data=state.model_dump(mode="json"),
        )

    def search_destination(self, query: str) -> ToolResult:
        """搜索目的地。"""
        results = self._provider.search(query)
        lines = [f"{i+1}. {r['name']}({r['address']}), 距离 {r['distance_km']} km" for i, r in enumerate(results)]
        output = "找到以下目的地:\n" + "\n".join(lines) if results else "未找到匹配的目的地。"
        return ToolResult(
            success=True,
            tool_name="search_destination",
            output=output,
            data={"results": results},
        )

    def start_navigation(self, destination: str) -> ToolResult:
        """开始导航。"""
        if not destination:
            return ToolResult(success=False, tool_name="start_navigation", output="未指定目的地。", error="missing_destination")
        self._provider.start(destination)
        output = f"已开始导航到 {destination}。"
        return ToolResult(success=True, tool_name="start_navigation", output=output, data={"destination": destination})

    def cancel_navigation(self) -> ToolResult:
        """取消导航。"""
        self._provider.cancel()
        return ToolResult(success=True, tool_name="cancel_navigation", output="导航已取消。")


NAVIGATION_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "get_navigation_status",
        "description": "查询当前导航状态,包括目的地、预计时间、剩余距离。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_destination",
        "description": "根据关键词搜索导航目的地。",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "start_navigation",
        "description": "开始导航到指定目的地。",
        "input_schema": {
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"],
        },
    },
    {
        "name": "cancel_navigation",
        "description": "取消当前导航。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def execute_navigation_tool(name: str, arguments: dict, service: NavigationService) -> ToolResult:
    """分派到 NavigationService 对应方法。"""
    if name == "get_navigation_status":
        return service.get_navigation_status()
    if name == "search_destination":
        return service.search_destination(arguments.get("query", ""))
    if name == "start_navigation":
        return service.start_navigation(arguments.get("destination", ""))
    if name == "cancel_navigation":
        return service.cancel_navigation()
    return ToolResult(success=False, tool_name=name, output="", error=f"unknown_tool:{name}")


_default_service: NavigationService | None = None


def get_navigation_service() -> NavigationService:
    """获取默认 NavigationService 单例。"""
    global _default_service
    if _default_service is None:
        _default_service = NavigationService(SimulatedNavigationProvider())
    return _default_service
