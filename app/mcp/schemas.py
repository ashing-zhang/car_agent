# MCP Tool Schemas - Automotive MCP Server 工具元数据(规格第11节)
# 运行指南:
#   每个工具含 name/description/inputSchema/outputSchema(规格第11节要求)
#   供 MCP tools/list 与单元测试校验使用
#   实际执行由 app/mcp/server.py 委托 ToolRegistry

MCP_TOOLS: list[dict] = [
    {
        "name": "vehicle.get_status",
        "description": "查询当前车辆状态,包括车速、电量、车内温度、空调状态、位置。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "outputSchema": {
            "type": "object",
            "properties": {
                "speed_kmh": {"type": "number"},
                "battery_percent": {"type": "number"},
                "cabin_temperature_c": {"type": "number"},
            },
        },
    },
    {
        "name": "vehicle.set_temperature",
        "description": "设置目标车内温度(摄氏度,范围 18-30)。",
        "inputSchema": {
            "type": "object",
            "properties": {"temperature_c": {"type": "number"}},
            "required": ["temperature_c"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"target_temperature_c": {"type": "number"}, "success": {"type": "boolean"}},
        },
    },
    {
        "name": "navigation.get_status",
        "description": "查询当前导航状态,包括目的地、预计时间、剩余距离。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "outputSchema": {
            "type": "object",
            "properties": {"active": {"type": "boolean"}, "destination": {"type": "string"}},
        },
    },
    {
        "name": "navigation.search",
        "description": "根据关键词搜索导航目的地。",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"results": {"type": "array"}},
        },
    },
    {
        "name": "navigation.start",
        "description": "开始导航到指定目的地。",
        "inputSchema": {
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"destination": {"type": "string"}, "started": {"type": "boolean"}},
        },
    },
    {
        "name": "media.play",
        "description": "播放指定播放列表或歌单。",
        "inputSchema": {
            "type": "object",
            "properties": {"playlist": {"type": "string"}},
            "required": ["playlist"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"playlist": {"type": "string"}},
        },
    },
    {
        "name": "media.pause",
        "description": "暂停当前媒体播放。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "outputSchema": {"type": "object", "properties": {"paused": {"type": "boolean"}}},
    },
    {
        "name": "media.volume",
        "description": "设置媒体音量(范围 0-40)。",
        "inputSchema": {
            "type": "object",
            "properties": {"level": {"type": "integer"}},
            "required": ["level"],
        },
        "outputSchema": {"type": "object", "properties": {"volume": {"type": "integer"}}},
    },
    {
        "name": "environment.get_context",
        "description": "获取当前环境上下文(天气、交通、路况,Phase 6 多模态填充)。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "outputSchema": {
            "type": "object",
            "properties": {
                "weather": {"type": "string"},
                "traffic_density": {"type": "string"},
            },
        },
    },
]


MCP_TOOL_NAME_TO_REGISTRY = {
    "vehicle.get_status": "get_vehicle_status",
    "vehicle.set_temperature": "set_temperature",
    "navigation.get_status": "get_navigation_status",
    "navigation.search": "search_destination",
    "navigation.start": "start_navigation",
    "media.play": "play_media",
    "media.pause": "pause_media",
    "media.volume": "set_volume",
}


def get_mcp_tool_names() -> list[str]:
    """返回全部 MCP 工具名。"""
    return [t["name"] for t in MCP_TOOLS]


def find_mcp_tool(name: str) -> dict | None:
    """按名称查找 MCP 工具元数据。"""
    for tool in MCP_TOOLS:
        if tool["name"] == name:
            return tool
    return None
