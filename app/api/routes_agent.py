# Agent API 路由 - POST /api/v1/agent/chat(规格第21节)
# 集成 Phase 8 可观测:每次请求记录 Trace + 运行 Metrics
# 运行指南:
#   curl -X POST http://localhost:8000/api/v1/agent/chat \
#     -H "Content-Type: application/json" \
#     -d '{"user_id":"demo-user","session_id":"session-001","message":"有点冷"}'

import logging
import time

from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.agent.graph import build_default_agent
from app.agent.plan_graph import run_plan
from app.config import get_settings
from app.observability.models import new_request_id
from app.observability.recorder import get_metrics_registry, get_trace_recorder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    """Agent chat 请求体(规格第21节)。"""

    user_id: str = "demo-user"
    session_id: str = "session-001"
    message: str


class ToolCallInfo(BaseModel):
    """单次工具调用信息。"""

    name: str
    arguments: dict


class ChatResponse(BaseModel):
    """Agent chat 响应体(规格第21节)。"""

    session_id: str
    request_id: str
    response: str
    tool_calls: list[ToolCallInfo]
    latency_ms: float


def _settings_model_name() -> str:
    """返回用于 Trace 记录的模型名。"""
    try:
        return get_settings().resolved_llm_model()
    except Exception:
        return "unknown"


def _finalize_tool_calls(raw_calls: list[ToolCallInfo]) -> list[dict]:
    """将 ToolCallInfo 列表转为 trace 可序列化 dict。"""
    return [tc.model_dump() for tc in raw_calls]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """处理用户消息并返回 Agent 回复与工具调用记录。"""
    request_id = new_request_id()
    start = time.perf_counter()
    trace_rec = get_trace_recorder()
    metrics = get_metrics_registry()
    trace = trace_rec.start_trace(
        session_id=req.session_id,
        user_id=req.user_id,
        model=_settings_model_name(),
        input_text=req.message,
        request_id=request_id,
    )

    tool_calls: list[ToolCallInfo] = []
    final_response = ""
    error_msg: str | None = None
    try:
        llm_span = trace_rec.start_span(trace, "agent_invoke", {"input": req.message})
        agent = build_default_agent()
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "user_id": req.user_id,
                "session_id": req.session_id,
            }
        )
        llm_span.finish()
        messages = result["messages"]
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(ToolCallInfo(name=tc["name"], arguments=tc["args"]))
            if isinstance(msg, AIMessage) and msg.content and msg is messages[-1]:
                final_response = msg.content if isinstance(msg.content, str) else str(msg.content)

        for tc in tool_calls:
            metrics.inc_counter("tool_calls_total", labels={"tool": tc.name})

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("agent chat failed request_id=%s", request_id)
        metrics.inc_counter("agent_runs_failed_total", labels={"status": "error"})
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        trace_rec.end_trace(
            trace,
            final_response=final_response,
            tool_calls=_finalize_tool_calls(tool_calls),
            error=error_msg,
            latency_ms=latency_ms,
        )
        metrics.inc_counter(
            "agent_runs_total",
            labels={"status": "error" if error_msg else "ok"},
        )
        metrics.observe_histogram(
            "agent_run_latency_ms",
            latency_ms,
            labels={"status": "error" if error_msg else "ok"},
        )
        logger.info(
            "agent.chat.done request_id=%s session=%s latency_ms=%.1f tool_calls=%d",
            request_id, req.session_id, latency_ms, len(tool_calls),
            extra={"request_id": request_id, "session_id": req.session_id, "latency_ms": latency_ms},
        )

    return ChatResponse(
        session_id=req.session_id,
        request_id=request_id,
        response=final_response,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
    )


@router.post("/plan", response_model=ChatResponse)
async def plan(req: ChatRequest) -> ChatResponse:
    """多步任务规划与执行(规格第16节,Phase 4)。"""
    request_id = new_request_id()
    start = time.perf_counter()
    trace_rec = get_trace_recorder()
    metrics = get_metrics_registry()
    trace = trace_rec.start_trace(
        session_id=req.session_id,
        user_id=req.user_id,
        model=_settings_model_name(),
        input_text=req.message,
        request_id=request_id,
    )

    tool_calls: list[ToolCallInfo] = []
    final_response = ""
    error_msg: str | None = None
    try:
        plan_span = trace_rec.start_span(trace, "plan_execute", {"input": req.message})
        state = run_plan(req.message, req.user_id)
        plan_span.finish()
        tool_calls = [
            ToolCallInfo(name=tc.name, arguments=tc.arguments)
            for tc in state.get("tool_calls", [])
        ]
        final_response = state.get("response", "")
        for tc in tool_calls:
            metrics.inc_counter("tool_calls_total", labels={"tool": tc.name})
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("agent plan failed request_id=%s", request_id)
        metrics.inc_counter("agent_runs_failed_total", labels={"status": "error"})
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        trace_rec.end_trace(
            trace,
            final_response=final_response,
            tool_calls=_finalize_tool_calls(tool_calls),
            error=error_msg,
            latency_ms=latency_ms,
        )
        metrics.inc_counter(
            "agent_runs_total",
            labels={"status": "error" if error_msg else "ok"},
        )
        metrics.observe_histogram(
            "agent_run_latency_ms",
            latency_ms,
            labels={"status": "error" if error_msg else "ok"},
        )

    return ChatResponse(
        session_id=req.session_id,
        request_id=request_id,
        response=final_response,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
    )
