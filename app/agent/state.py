# LangGraph Agent State - 显式状态定义(规格第7节)
# 运行指南: 由 app/agent/graph.py 各节点读写,禁止依赖隐式全局变量

from typing import Any, TypedDict

from app.context.schemas import EnvironmentState, NavigationState, VehicleState
from app.tools.schemas import PlanStep, ToolCallRecord, ToolResult


class AgentState(TypedDict, total=False):
    """LangGraph 状态容器。"""

    session_id: str
    user_id: str

    user_message: str

    vehicle_state: VehicleState | None
    environment_state: EnvironmentState | None
    navigation_state: NavigationState | None

    memories: list[Any]  # Phase 3 填充为 list[Memory]
    plan: list[PlanStep]

    tool_calls: list[ToolCallRecord]
    tool_results: list[ToolResult]

    response: str

    error: str | None
