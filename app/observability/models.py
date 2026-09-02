# Observability 数据模型 - Trace/Span/RunSummary(规格第20节,Phase 8)
# 运行指南:
#   from app.observability.models import Trace, Span
#   span = Span(name="llm_call", latency_ms=120.0)
#   trace = Trace(request_id="req-1", session_id="s-1", user_id="u-1")

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


def _now_ms() -> float:
    """返回当前 epoch 毫秒时间戳。"""
    return time.time() * 1000.0


def new_request_id() -> str:
    """生成短型 request_id。"""
    return f"req-{uuid.uuid4().hex[:12]}"


class Span(BaseModel):
    """单个 Trace Span,对应一次子操作(LLM/Tool/Memory)。"""

    span_id: str
    name: str
    start_ms: float
    end_ms: float | None = None
    latency_ms: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @classmethod
    def start(cls, name: str, attributes: dict[str, Any] | None = None) -> "Span":
        """开始一个 Span,记录起始时间。"""
        return cls(
            span_id=uuid.uuid4().hex[:16],
            name=name,
            start_ms=_now_ms(),
            attributes=dict(attributes or {}),
        )

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        """结束 Span 并计算延迟。"""
        self.end_ms = _now_ms()
        self.latency_ms = self.end_ms - self.start_ms
        self.status = status
        self.error = error


class Trace(BaseModel):
    """单次 Agent Run 的完整 Trace(规格第20节)。"""

    request_id: str
    session_id: str
    user_id: str
    model: str
    prompt_version: str = "v1"
    input: str
    context: dict[str, Any] = Field(default_factory=dict)
    plan: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)
    memory_reads: int = 0
    memory_writes: int = 0
    final_response: str = ""
    error: str | None = None
    spans: list[Span] = Field(default_factory=list)
    created_at_ms: float = Field(default_factory=_now_ms)


class RunSummary(BaseModel):
    """Agent Run 的精简摘要(Dashboard 列表用)。"""

    request_id: str
    session_id: str
    user_id: str
    input: str
    final_response: str
    success: bool
    latency_ms: float
    tool_call_count: int
    error: str | None = None
    created_at_ms: float
