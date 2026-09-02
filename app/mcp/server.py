# Automotive MCP Server - 暴露 vehicle/navigation/media 工具(规格第11节)
# 运行指南:
#   stdio 模式: python -m app.mcp.server
#   MCP Client 通过 tools/list 发现工具,通过 tool call 调用
#   遵循 MCP 规范,使用 FastMCP 真实实现 tools/list 与 tool call
#   mcp 包未安装时降级(不影响其他模块)

import logging

from app.agent.tool_registry import get_tool_registry

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp 未安装时降级,不影响其他模块
    FastMCP = None  # type: ignore[assignment,misc]


def build_server():
    """构造 FastMCP 实例并注册全部 Automotive 工具(规格第11节)。"""
    if FastMCP is None:
        logger.warning("mcp package not installed; MCP server unavailable")
        return None
    mcp = FastMCP("AutoAgent-MCP")
    registry = get_tool_registry()

    @mcp.tool(name="vehicle.get_status", description="查询当前车辆状态,包括车速、电量、车内温度、空调状态、位置。")
    def vehicle_get_status() -> str:
        """查询当前车辆状态。"""
        return registry.execute("get_vehicle_status", {})

    @mcp.tool(name="vehicle.set_temperature", description="设置目标车内温度(摄氏度,范围 18-30)。")
    def vehicle_set_temperature(temperature_c: float) -> str:
        """设置目标车内温度。"""
        return registry.execute("set_temperature", {"temperature_c": temperature_c})

    @mcp.tool(name="navigation.get_status", description="查询当前导航状态,包括目的地、预计时间、剩余距离。")
    def navigation_get_status() -> str:
        """查询当前导航状态。"""
        return registry.execute("get_navigation_status", {})

    @mcp.tool(name="navigation.search", description="根据关键词搜索导航目的地。")
    def navigation_search(query: str) -> str:
        """搜索导航目的地。"""
        return registry.execute("search_destination", {"query": query})

    @mcp.tool(name="navigation.start", description="开始导航到指定目的地。")
    def navigation_start(destination: str) -> str:
        """开始导航到指定目的地。"""
        return registry.execute("start_navigation", {"destination": destination})

    @mcp.tool(name="media.play", description="播放指定播放列表或歌单。")
    def media_play(playlist: str) -> str:
        """播放指定播放列表。"""
        return registry.execute("play_media", {"playlist": playlist})

    @mcp.tool(name="media.pause", description="暂停当前媒体播放。")
    def media_pause() -> str:
        """暂停媒体播放。"""
        return registry.execute("pause_media", {})

    @mcp.tool(name="media.volume", description="设置媒体音量(范围 0-40)。")
    def media_volume(level: int) -> str:
        """设置媒体音量。"""
        return registry.execute("set_volume", {"level": level})

    @mcp.tool(name="environment.get_context", description="获取当前环境上下文(天气、交通、路况)。")
    def environment_get_context() -> str:
        """获取环境上下文(场景池随机选景提供)。"""
        return registry.execute("get_camera_scene", {})

    return mcp


mcp = build_server()


def main() -> None:
    """启动 MCP Server(stdio transport)。"""
    if mcp is None:
        logger.error("mcp package not installed; run: pip install mcp")
        return
    logging.basicConfig(level=logging.INFO)
    logger.info("AutoAgent MCP Server starting (stdio transport)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
