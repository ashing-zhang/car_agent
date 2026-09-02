# MCP Schemas 单元测试(规格第11节,Phase 5)
# 运行指南: pytest tests/unit/test_mcp.py -v
# 不依赖 mcp 包,验证工具元数据与映射逻辑

from app.agent.tool_registry import get_tool_registry
from app.mcp.schemas import MCP_TOOLS, MCP_TOOL_NAME_TO_REGISTRY, find_mcp_tool, get_mcp_tool_names


def test_mcp_tools_count_meets_spec() -> None:
    """验证 MCP 工具数量覆盖规格第11节全部能力。"""
    names = get_mcp_tool_names()
    assert "vehicle.get_status" in names
    assert "vehicle.set_temperature" in names
    assert "navigation.get_status" in names
    assert "navigation.search" in names
    assert "navigation.start" in names
    assert "media.play" in names
    assert "media.pause" in names
    assert "media.volume" in names
    assert "environment.get_context" in names


def test_each_tool_has_required_fields() -> None:
    """验证每个工具含 name/description/inputSchema/outputSchema(规格第11节)。"""
    for tool in MCP_TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert "outputSchema" in tool
        assert tool["description"], f"工具 {tool['name']} 缺少描述"


def test_find_mcp_tool() -> None:
    """验证按名称查找工具元数据。"""
    tool = find_mcp_tool("vehicle.set_temperature")
    assert tool is not None
    assert "temperature_c" in tool["inputSchema"]["properties"]


def test_registry_mapping_executes_vehicle_status() -> None:
    """验证 MCP 工具名映射后可通过 ToolRegistry 执行。"""
    registry_name = MCP_TOOL_NAME_TO_REGISTRY["vehicle.get_status"]
    output = get_tool_registry().execute(registry_name, {})
    assert "车速" in output


def test_registry_mapping_executes_set_temperature() -> None:
    """验证 vehicle.set_temperature 映射执行成功。"""
    registry_name = MCP_TOOL_NAME_TO_REGISTRY["vehicle.set_temperature"]
    output = get_tool_registry().execute(registry_name, {"temperature_c": 24})
    assert "24" in output


def test_registry_mapping_executes_navigation_search() -> None:
    """验证 navigation.search 映射执行成功。"""
    registry_name = MCP_TOOL_NAME_TO_REGISTRY["navigation.search"]
    output = get_tool_registry().execute(registry_name, {"query": "公司"})
    assert "公司" in output
