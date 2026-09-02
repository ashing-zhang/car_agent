# LangGraph Agent 图 - ReAct + Memory 集成(规格第15节,Phase 2-3)
# 运行指南:
#   agent = build_default_agent()
#   result = agent.invoke({"messages":[HumanMessage("有点冷")], "user_id":"demo-user"})
#   循环: retrieve_memory → agent ↔ tools → extract_memory → END
# Phase 4 将加入 plan 节点扩展为完整图

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.tools import build_vehicle_tools
from app.agent.tool_registry import ToolRegistry, get_tool_registry
from app.llm.provider import LLMProvider, get_llm_provider
from app.memory.service import get_memory_service
from app.tools.vehicle import VehicleService, get_vehicle_service

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "system.md").read_text(encoding="utf-8")


class AgentGraphState(TypedDict, total=False):
    """ReAct Agent 状态,含 messages 与会话标识。"""

    messages: list[BaseMessage]
    user_id: str
    session_id: str


def _last_human_content(messages: list[BaseMessage]) -> str:
    """提取最后一条用户消息文本。"""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _make_retrieve_memory_node(memory_service: object | None = None) -> object:
    """构造记忆检索节点:注入用户偏好到上下文。"""

    def node(state: AgentGraphState) -> dict:
        """检索用户偏好并注入为 SystemMessage。"""
        user_id = state.get("user_id", "demo-user")
        service = memory_service if memory_service is not None else get_memory_service()
        pref = service.recall_preference(user_id, "preferred_temperature")
        if pref is None:
            return {}
        ctx = f"已知用户偏好:车内温度 {pref.value}℃。若用户表达冷热感受,优先采用该偏好温度。"
        return {"messages": [SystemMessage(content=ctx)]}

    return node


def _make_extract_memory_node(memory_service: object | None = None) -> object:
    """构造记忆提取节点:从用户消息提取并持久化偏好。"""

    def node(state: AgentGraphState) -> dict:
        """从最后一条用户消息提取记忆。"""
        user_id = state.get("user_id", "demo-user")
        session_id = state.get("session_id", "session-001")
        service = memory_service if memory_service is not None else get_memory_service()
        user_text = _last_human_content(state.get("messages", []))
        if user_text:
            service.remember(user_text, user_id, session_id)
        return {}

    return node


def _make_agent_node(llm: object) -> object:
    """构造 agent 节点:注入 system prompt 后调用 LLM。"""

    def node(state: AgentGraphState) -> dict:
        """调用 LLM 决定调用工具或生成回复。"""
        messages: list[BaseMessage] = list(state.get("messages", []))
        if not any(isinstance(m, SystemMessage) and "AutoAgent" in (m.content or "") for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm.invoke(messages)
        return {"messages": [response]}

    return node


def build_agent(
    service: VehicleService,
    llm_provider: LLMProvider,
    memory_service: object | None = None,
    registry: ToolRegistry | None = None,
) -> object:
    """构建含记忆集成的 LangGraph Agent,memory_service 与 registry 可注入。"""
    if registry is not None:
        tools = registry.tools
    else:
        tools = build_vehicle_tools(service)
    llm_with_tools = llm_provider.bind_tools(tools)
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentGraphState)
    workflow.add_node("retrieve_memory", _make_retrieve_memory_node(memory_service))
    workflow.add_node("agent", _make_agent_node(llm_with_tools))
    workflow.add_node("tools", tool_node)
    workflow.add_node("extract_memory", _make_extract_memory_node(memory_service))

    workflow.add_edge(START, "retrieve_memory")
    workflow.add_edge("retrieve_memory", "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "extract_memory"})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("extract_memory", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def build_default_agent() -> object:
    """使用默认 service、LLM provider 与全工具 registry 构建单例 Agent。"""
    service = get_vehicle_service()
    provider = get_llm_provider()
    registry = get_tool_registry()
    return build_agent(service, provider, registry=registry)
