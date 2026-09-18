# 意图分类 Domain 模型 - 定义 Agent 类型与分类结果数据结构
# 运行指南:
#   from app.agent.intent.models import AgentType, IntentClassification
#   cls = IntentClassification(agent_type=AgentType.PLAN_EXECUTE, confidence=0.95, reason="multi-step navigation")

from enum import StrEnum

from pydantic import BaseModel, Field


class AgentType(StrEnum):
    """Agent 执行模式枚举。"""

    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


class IntentClassification(BaseModel):
    """意图分类结果:决定使用 ReAct 还是 Plan-Execute。"""

    agent_type: AgentType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    signals: dict[str, int] = Field(default_factory=dict)
