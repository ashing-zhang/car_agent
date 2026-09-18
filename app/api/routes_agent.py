# Agent API 路由 - POST /api/v1/agent/chat 统一入口(规格第21节)
# 集成 Phase 8 可观测 + Phase 9 意图分类自动路由
# 运行指南:
#   curl -X POST http://localhost:8000/api/v1/agent/chat \
#     -H "Content-Type: application/json" \
#     -d '{"user_id":"demo-user","session_id":"session-001","message":"有点冷"}'
#   多步任务(如"去公司顺便播放音乐")会自动路由到 Plan-Execute,无需用户指定

import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.intent.models import AgentType
from app.agent.orchestrator import AgentOrchestrator, build_default_orchestrator
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
    agent_type: AgentType
    classification_confidence: float
    classification_reason: str


def _settings_model_name() -> str:
    """返回用于 Trace 记录的模型名。"""
    try:
        return get_settings().resolved_llm_model()
    except Exception:
        return "unknown"


def _finalize_tool_calls(raw_calls: list[ToolCallInfo]) -> list[dict]:
    """将 ToolCallInfo 列表转为 trace 可序列化 dict。"""
    return [tc.model_dump() for tc in raw_calls]


async def _execute_chat(
    req: ChatRequest,
    orchestrator: AgentOrchestrator | None = None,
) -> ChatResponse:
    """统一执行聊天:通过编排器自动分类并路由 Agent,对外屏蔽内部差异。"""
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
    agent_type: AgentType = AgentType.REACT
    conf: float = 0.0
    reason: str = ""
    try:
        orch = orchestrator or build_default_orchestrator()
        llm_span = trace_rec.start_span(trace, "orchestrator_run", {"input": req.message})
        result = orch.run(req.message, user_id=req.user_id, session_id=req.session_id)
        llm_span.finish()

        final_response = result.get("response", "")
        raw_tcs = result.get("tool_calls", [])
        tool_calls = [
            ToolCallInfo(name=tc.get("name", ""), arguments=tc.get("arguments", {}))
            for tc in raw_tcs
        ]
        agent_type = result.get("agent_type", AgentType.REACT)
        cls = result.get("classification")
        if cls is not None:
            conf = cls.confidence
            reason = cls.reason

        for tc in tool_calls:
            metrics.inc_counter("tool_calls_total", labels={"tool": tc.name})

        metrics.inc_counter(
            "intent_classification_total",
            labels={"agent_type": str(agent_type)},
        )

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
            "agent.chat.done request_id=%s session=%s latency_ms=%.1f tools=%d type=%s conf=%.2f",
            request_id, req.session_id, latency_ms, len(tool_calls), agent_type, conf,
            extra={
                "request_id": request_id,
                "session_id": req.session_id,
                "latency_ms": latency_ms,
                "agent_type": str(agent_type),
                "classification_confidence": conf,
            },
        )

    return ChatResponse(
        session_id=req.session_id,
        request_id=request_id,
        response=final_response,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
        agent_type=agent_type,
        classification_confidence=conf,
        classification_reason=reason,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """处理用户消息并返回 Agent 回复(自动识别单步/多步意图)。"""
    return await _execute_chat(req)
