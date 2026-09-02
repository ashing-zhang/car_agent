# 工具注册表 - 统一管理 vehicle/navigation/media 工具(规格第9节)
# 运行指南:
#   registry = get_tool_registry()
#   for tool in registry.tools: ...  # 用于 LLM bind_tools
#   result = registry.execute("search_destination", {"query":"公司"})

import logging
from typing import Any

from langchain.tools import tool

from app.tools.environment import EnvironmentService, get_environment_service
from app.tools.media import MediaService, get_media_service
from app.tools.navigation import NavigationService, get_navigation_service
from app.tools.vehicle import VehicleService, get_vehicle_service

logger = logging.getLogger(__name__)


class ToolRegistry:
    """所有工具的注册中心,供 LLM bind_tools 与 plan-execute 执行器共用。"""

    def __init__(
        self,
        vehicle: VehicleService,
        navigation: NavigationService,
        media: MediaService,
        environment: EnvironmentService | None = None,
    ) -> None:
        """注入四类服务并构建工具。"""
        self._vehicle = vehicle
        self._navigation = navigation
        self._media = media
        self._environment = environment or get_environment_service()
        self._tools: list[Any] = self._build_tools()
        self._by_name: dict[str, Any] = {t.name: t for t in self._tools}

    def _build_tools(self) -> list[Any]:
        """构造绑定到各 service 的 @tool 函数列表。"""

        @tool
        def get_vehicle_status() -> str:
            """查询当前车辆状态,包括车速、电量、车内温度、空调状态、位置等。"""
            return self._vehicle.get_vehicle_status().output

        @tool
        def get_cabin_temperature() -> str:
            """查询当前车内温度。"""
            return self._vehicle.get_cabin_temperature().output

        @tool
        def set_temperature(temperature_c: float) -> str:
            """设置目标车内温度(摄氏度,范围 18-30)。"""
            return self._vehicle.set_temperature(temperature_c).output

        @tool
        def set_ac(enabled: bool) -> str:
            """开启或关闭空调。"""
            return self._vehicle.set_ac(enabled).output

        @tool
        def get_navigation_status() -> str:
            """查询当前导航状态,包括目的地、预计时间、剩余距离。"""
            return self._navigation.get_navigation_status().output

        @tool
        def search_destination(query: str) -> str:
            """根据关键词搜索导航目的地。"""
            return self._navigation.search_destination(query).output

        @tool
        def start_navigation(destination: str) -> str:
            """开始导航到指定目的地。"""
            return self._navigation.start_navigation(destination).output

        @tool
        def cancel_navigation() -> str:
            """取消当前导航。"""
            return self._navigation.cancel_navigation().output

        @tool
        def play_media(playlist: str) -> str:
            """播放指定播放列表或歌单。"""
            return self._media.play_media(playlist).output

        @tool
        def pause_media() -> str:
            """暂停当前媒体播放。"""
            return self._media.pause_media().output

        @tool
        def set_volume(level: int) -> str:
            """设置媒体音量(范围 0-40)。"""
            return self._media.set_volume(level).output

        @tool
        def get_weather() -> str:
            """查询当前天气状况(基于场景池随机选景)。"""
            return self._environment.get_weather().output

        @tool
        def get_traffic() -> str:
            """查询当前交通密度。"""
            return self._environment.get_traffic().output

        @tool
        def get_camera_scene() -> str:
            """返回当前场景的完整结构化解析(天气/交通/施工/行人)。"""
            return self._environment.get_camera_scene().output

        return [
            get_vehicle_status, get_cabin_temperature, set_temperature, set_ac,
            get_navigation_status, search_destination, start_navigation, cancel_navigation,
            play_media, pause_media, set_volume,
            get_weather, get_traffic, get_camera_scene,
        ]

    @property
    def tools(self) -> list[Any]:
        """返回全部工具(供 bind_tools)。"""
        return self._tools

    @property
    def names(self) -> list[str]:
        """返回全部工具名。"""
        return list(self._by_name.keys())

    def execute(self, name: str, arguments: dict) -> str:
        """按名称执行工具并返回输出文本。"""
        tool_func = self._by_name.get(name)
        if tool_func is None:
            logger.warning("Unknown tool requested: %s", name)
            return f"未知工具: {name}"
        try:
            return tool_func.invoke(arguments)
        except Exception as exc:
            logger.exception("Tool %s execution failed", name)
            return f"工具执行失败: {name}({exc})"


_default_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """获取默认 ToolRegistry 单例。"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry(
            get_vehicle_service(),
            get_navigation_service(),
            get_media_service(),
        )
    return _default_registry
