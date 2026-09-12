# Observability Recorder - Trace 收集器 + 运行时 Metrics(规格第19/20节,Phase 8)
# 运行指南:
#   from app.observability.recorder import get_trace_recorder, get_metrics_registry
#   tr = get_trace_recorder()
#   trace = tr.start_trace(...)
#   tr.end_trace(trace, ...)
#   mr = get_metrics_registry()
#   mr.inc_counter("agent_runs_total", labels={"status":"ok"})

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

from app.config import get_app_config
from app.observability.models import RunSummary, Span, Trace, new_request_id

logger = logging.getLogger(__name__)


# ---------------- Metrics primitives ----------------


class Counter:
    """单调递增计数器。"""

    def __init__(self, name: str, help_text: str) -> None:
        """初始化计数器。"""
        self.name = name
        self.help_text = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """增量增加。"""
        key = _labels_key(labels)
        self._values[key] += value

    def samples(self) -> list[tuple[dict[str, str], float]]:
        """返回 (labels, value) 样本列表。"""
        return [(dict(k), v) for k, v in self._values.items()]


class Histogram:
    """延迟直方图,带可配置 bucket 边界。"""

    def __init__(self, name: str, help_text: str, buckets: list[float]) -> None:
        """初始化直方图。"""
        self.name = name
        self.help_text = help_text
        self.buckets = sorted(buckets)
        self._counts: dict[tuple[tuple[str, str], ...], list[int]] = defaultdict(
            lambda: [0] * (len(self.buckets) + 1)
        )
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._totals: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """记录一次观测值。"""
        key = _labels_key(labels)
        counts = self._counts[key]
        idx = 0
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                idx = i
                break
        else:
            idx = len(self.buckets)
        for j in range(idx, len(counts)):
            counts[j] += 1
        self._sums[key] += value
        self._totals[key] += 1

    def samples(self) -> list[tuple[dict[str, str], list[int], float, int]]:
        """返回 (labels, cumulative_counts, sum, total) 样本列表。"""
        return [
            (dict(k), list(self._counts[k]), self._sums[k], self._totals[k])
            for k in self._counts
        ]


def _labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    """将 labels dict 转为可哈希的有序 key。"""
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


# ---------------- Metrics Registry ----------------


class MetricsRegistry:
    """运行时 Metrics 注册中心,管理 Counter/Histogram。"""

    def __init__(self) -> None:
        """初始化空注册中心。"""
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        """注册默认指标。"""
        cfg = get_app_config().observability
        self.counter("agent_runs_total", "Total agent runs")
        self.counter("agent_runs_failed_total", "Total failed agent runs")
        self.counter("tool_calls_total", "Total tool calls invoked")
        self.counter("memory_reads_total", "Total memory read operations")
        self.counter("memory_writes_total", "Total memory write operations")
        self.counter("llm_calls_total", "Total LLM invocations")
        self.histogram(
            "agent_run_latency_ms",
            "Agent run latency in milliseconds",
            cfg.latency_buckets_ms,
        )
        self.histogram(
            "tool_call_latency_ms",
            "Tool call latency in milliseconds",
            cfg.latency_buckets_ms,
        )

    def counter(self, name: str, help_text: str) -> Counter:
        """获取或注册计数器。"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, help_text)
            return self._counters[name]

    def histogram(self, name: str, help_text: str, buckets: list[float]) -> Histogram:
        """获取或注册直方图。"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, help_text, buckets)
            return self._histograms[name]

    def inc_counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """便捷计数器增量。"""
        if name in self._counters:
            self._counters[name].inc(value, labels)
        else:
            logger.warning("Counter %s not registered", name)

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """便捷直方图观测。"""
        if name in self._histograms:
            self._histograms[name].observe(value, labels)
        else:
            logger.warning("Histogram %s not registered", name)

    def counters(self) -> dict[str, Counter]:
        """返回全部计数器。"""
        return dict(self._counters)

    def histograms(self) -> dict[str, Histogram]:
        """返回全部直方图。"""
        return dict(self._histograms)

    def snapshot(self) -> dict[str, Any]:
        """生成指标快照(Dashboard 展示用)。"""
        counters_snap = {
            n: [{"labels": c[0], "value": c[1]} for c in cnt.samples()]
            for n, cnt in self._counters.items()
        }
        histograms_snap = {
            n: [
                {
                    "labels": s[0],
                    "buckets": s[1],
                    "sum": s[2],
                    "total": s[3],
                    "bucket_bounds": list(self._histograms[n].buckets),
                }
                for s in hist.samples()
            ]
            for n, hist in self._histograms.items()
        }
        return {"counters": counters_snap, "histograms": histograms_snap}


# ---------------- Trace Recorder ----------------


class TraceRecorder:
    """Trace 收集器,内存环形缓冲,提供 Dashboard 查询。"""

    def __init__(self, store_limit: int = 200) -> None:
        """初始化空收集器与容量上限。"""
        self._traces: list[Trace] = []
        self._by_id: dict[str, int] = {}
        self._by_user: dict[str, list[int]] = defaultdict(list)
        self._by_session: dict[str, list[int]] = defaultdict(list)
        self._store_limit = store_limit
        self._lock = threading.Lock()

    def start_trace(
        self,
        session_id: str,
        user_id: str,
        model: str,
        input_text: str,
        prompt_version: str = "v1",
        request_id: str | None = None,
    ) -> Trace:
        """开始一个新的 Agent Run Trace。"""
        trace = Trace(
            request_id=request_id or new_request_id(),
            session_id=session_id,
            user_id=user_id,
            model=model,
            prompt_version=prompt_version,
            input=input_text,
        )
        logger.info(
            "trace.start request_id=%s session=%s user=%s",
            trace.request_id, session_id, user_id,
        )
        return trace

    def start_span(self, trace: Trace, name: str, attributes: dict[str, Any] | None = None) -> Span:
        """在 trace 上开始一个 span 并返回引用。"""
        span = Span.start(name, attributes)
        trace.spans.append(span)
        return span

    def end_trace(
        self,
        trace: Trace,
        final_response: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        memory_reads: int = 0,
        memory_writes: int = 0,
        token_usage: dict[str, int] | None = None,
        error: str | None = None,
        latency_ms: float | None = None,
    ) -> Trace:
        """结束 trace 并存入缓冲。"""
        trace.final_response = final_response
        trace.tool_calls = tool_calls or []
        trace.tool_results = tool_results or []
        trace.memory_reads = memory_reads
        trace.memory_writes = memory_writes
        trace.token_usage = token_usage or {}
        trace.error = error
        trace.latency_ms = latency_ms if latency_ms is not None else trace.latency_ms
        self._store(trace)
        return trace

    def _store(self, trace: Trace) -> None:
        """以环形缓冲方式存入 trace。"""
        with self._lock:
            if trace.request_id in self._by_id:
                idx = self._by_id[trace.request_id]
                self._traces[idx] = trace
            else:
                if len(self._traces) >= self._store_limit:
                    oldest = self._traces.pop(0)
                    self._by_id.pop(oldest.request_id, None)
                    self._remove_from_secondary_index(oldest.user_id, oldest.session_id, 0)
                    self._by_id = {k: v - 1 for k, v in self._by_id.items()}
                    self._shift_secondary_indexes()
                new_idx = len(self._traces)
                self._by_id[trace.request_id] = new_idx
                self._by_user[trace.user_id].append(new_idx)
                self._by_session[trace.session_id].append(new_idx)
                self._traces.append(trace)

    def _remove_from_secondary_index(self, user_id: str, session_id: str, idx: int) -> None:
        """从二级索引中移除指定位置的记录(内部调用,需持有锁)。"""
        user_list = self._by_user.get(user_id)
        if user_list:
            user_list[:] = [i for i in user_list if i != idx]
            if not user_list:
                self._by_user.pop(user_id, None)
        session_list = self._by_session.get(session_id)
        if session_list:
            session_list[:] = [i for i in session_list if i != idx]
            if not session_list:
                self._by_session.pop(session_id, None)

    def _shift_secondary_indexes(self) -> None:
        """环形缓冲淘汰首条后,所有位置-1(内部调用,需持有锁)。"""
        for uid in list(self._by_user.keys()):
            self._by_user[uid] = [i - 1 for i in self._by_user[uid] if i > 0]
            if not self._by_user[uid]:
                self._by_user.pop(uid, None)
        for sid in list(self._by_session.keys()):
            self._by_session[sid] = [i - 1 for i in self._by_session[sid] if i > 0]
            if not self._by_session[sid]:
                self._by_session.pop(sid, None)

    def by_user(self, user_id: str, limit: int = 20) -> list[Trace]:
        """按 user_id 查询该用户最近 N 条 trace(按时间倒序)。"""
        with self._lock:
            idxs = self._by_user.get(user_id, [])
            return [self._traces[i] for i in reversed(idxs[-limit:])]

    def by_session(self, session_id: str, limit: int = 20) -> list[Trace]:
        """按 session_id 查询该会话最近 N 条 trace(按时间倒序)。"""
        with self._lock:
            idxs = self._by_session.get(session_id, [])
            return [self._traces[i] for i in reversed(idxs[-limit:])]

    def get(self, request_id: str) -> Trace | None:
        """按 request_id 查询 trace。"""
        with self._lock:
            idx = self._by_id.get(request_id)
            return self._traces[idx] if idx is not None else None

    def recent(self, limit: int = 20) -> list[Trace]:
        """返回最近 N 条 trace(按时间倒序)。"""
        with self._lock:
            return list(reversed(self._traces[-limit:]))

    def summaries(self, limit: int = 50) -> list[RunSummary]:
        """返回最近 trace 的摘要列表。"""
        traces = self.recent(limit)
        return [
            RunSummary(
                request_id=t.request_id,
                session_id=t.session_id,
                user_id=t.user_id,
                input=t.input,
                final_response=t.final_response,
                success=t.error is None,
                latency_ms=t.latency_ms,
                tool_call_count=len(t.tool_calls),
                error=t.error,
                created_at_ms=t.created_at_ms,
            )
            for t in traces
        ]


# ---------------- Singletons ----------------


_trace_recorder: TraceRecorder | None = None
_metrics_registry: MetricsRegistry | None = None
_singleton_lock = threading.Lock()


def get_trace_recorder() -> TraceRecorder:
    """获取 TraceRecorder 单例(线程安全)。"""
    global _trace_recorder
    if _trace_recorder is not None:
        return _trace_recorder
    with _singleton_lock:
        if _trace_recorder is None:
            cfg = get_app_config().observability
            _trace_recorder = TraceRecorder(store_limit=cfg.trace_store_limit)
    return _trace_recorder


def get_metrics_registry() -> MetricsRegistry:
    """获取 MetricsRegistry 单例(线程安全)。"""
    global _metrics_registry
    if _metrics_registry is not None:
        return _metrics_registry
    with _singleton_lock:
        if _metrics_registry is None:
            _metrics_registry = MetricsRegistry()
    return _metrics_registry
