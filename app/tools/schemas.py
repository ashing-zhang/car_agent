# 工具调用与结果 Schema
# 运行指南: Agent 工具执行后统一返回 ToolResult,便于 Observability 记录

from pydantic import BaseModel


class ToolCallRecord(BaseModel):
    """单次工具调用记录。"""

    name: str
    arguments: dict


class ToolResult(BaseModel):
    """工具执行结果,统一返回结构。"""

    success: bool
    tool_name: str
    output: str
    data: dict | None = None
    error: str | None = None


class PlanStep(BaseModel):
    """Planner 输出的单个步骤(规格第16节)。"""

    id: int
    tool: str
    arguments: dict
