# Plan-Execute Agent 图 - 多步任务规划与执行(规格第15、16节,Phase 4)
# 运行指南:
#   agent = build_plan_agent()
#   result = agent.invoke({"messages":[HumanMessage("去公司顺便播放音乐")], "user_id":"u1"})
#   流程: planner → execute(循环) → respond → END
#   输出含 tool_calls/tool_results/response,可展示完整 Plan

from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agent.planner import Planner, get_planner
from app.agent.tool_registry import ToolRegistry, get_tool_registry
from app.tools.schemas import ToolCallRecord, PlanStep, ToolResult


class PlanExecuteState(TypedDict, total=False):
    """plan-execute 状态容器。"""

    messages: list[BaseMessage]
    user_id: str
    session_id: str
    plan: list[PlanStep]
    tool_calls: list[ToolCallRecord]
    tool_results: list[ToolResult]
    response: str


def _last_human_content(messages: list[BaseMessage]) -> str:
    """提取最后一条用户消息文本。"""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _make_planner_node(planner: Planner) -> object:
    """构造规划节点:生成有序 PlanStep 列表。"""

    def node(state: PlanExecuteState) -> dict:
        """根据用户消息生成执行计划。"""
        user_text = _last_human_content(state.get("messages", []))
        user_id = state.get("user_id", "demo-user")
        plan = planner.plan(user_text, user_id)
        return {"plan": plan}

    return node


def _make_execute_node(registry: ToolRegistry) -> object:
    """构造执行节点:执行 plan 的下一步并记录结果。"""

    def node(state: PlanExecuteState) -> dict:
        """执行首条 step,移出 plan 并追加 tool 记录。"""
        plan: list[PlanStep] = list(state.get("plan", []))
        if not plan:
            return {}
        step = plan[0]
        output = registry.execute(step.tool, step.arguments)
        return {
            "plan": plan[1:],
            "tool_calls": list(state.get("tool_calls", []))
            + [ToolCallRecord(name=step.tool, arguments=step.arguments)],
            "tool_results": list(state.get("tool_results", []))
            + [ToolResult(success=True, tool_name=step.tool, output=output)],
        }

    return node


def _should_continue(state: PlanExecuteState) -> str:
    """条件路由:plan 非空继续执行,否则进入回复。"""
    return "execute" if state.get("plan") else "respond"


def _make_respond_node() -> object:
    """构造回复节点:汇总工具执行结果。"""

    def node(state: PlanExecuteState) -> dict:
        """拼接所有工具输出作为最终回复。"""
        results: list[ToolResult] = state.get("tool_results", [])
        if not results:
            return {"response": "我已了解您的需求,但未触发具体操作。"}
        summary = "\n".join(f"- {r.tool_name}: {r.output}" for r in results)
        return {"response": f"已为您完成以下操作:\n{summary}"}

    return node


def build_plan_agent(
    planner: Planner | None = None,
    registry: ToolRegistry | None = None,
) -> object:
    """构建 plan-execute Agent,planner 与 registry 可注入。"""
    planner = planner or get_planner()
    registry = registry or get_tool_registry()

    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", _make_planner_node(planner))
    workflow.add_node("execute", _make_execute_node(registry))
    workflow.add_node("respond", _make_respond_node())

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "execute")
    workflow.add_conditional_edges("execute", _should_continue, {"execute": "execute", "respond": "respond"})
    workflow.add_edge("respond", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def build_default_plan_agent() -> object:
    """使用默认 planner 与 registry 构建单例 plan-execute Agent。"""
    return build_plan_agent()


def run_plan(user_message: str, user_id: str = "demo-user") -> PlanExecuteState:
    """便捷执行:输入用户消息,返回完整 plan-execute 状态。"""
    agent = build_default_plan_agent()
    return agent.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
            "session_id": "plan-session",
        }
    )
