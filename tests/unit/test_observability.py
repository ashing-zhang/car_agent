# Phase 8 Observability 单元测试(规格第19/20节)
# 运行指南: pytest tests/unit/test_observability.py -v

from fastapi.testclient import TestClient

from app.main import app
from app.observability.models import Span, Trace, new_request_id
from app.observability.prometheus import render_prometheus
from app.observability.recorder import (
    MetricsRegistry,
    TraceRecorder,
    get_metrics_registry,
    get_trace_recorder,
)


# ---------------- Models ----------------


def test_span_lifecycle_records_latency() -> None:
    """Span start/finish 应计算 latency。"""
    span = Span.start("llm_call", {"model": "qwen"})
    assert span.name == "llm_call"
    assert span.end_ms is None
    span.finish()
    assert span.end_ms is not None
    assert span.latency_ms >= 0.0


def test_new_request_id_is_unique() -> None:
    """request_id 生成应唯一。"""
    a, b = new_request_id(), new_request_id()
    assert a != b
    assert a.startswith("req-")


# ---------------- Trace Recorder ----------------


def test_trace_recorder_stores_and_retrieves() -> None:
    """TraceRecorder 应能存取 trace。"""
    rec = TraceRecorder(store_limit=5)
    trace = rec.start_trace(
        session_id="s1", user_id="u1", model="qwen-plus", input_text="有点冷",
    )
    span = rec.start_span(trace, "agent_invoke")
    span.finish()
    rec.end_trace(
        trace,
        final_response="已调温",
        tool_calls=[{"name": "set_temperature", "arguments": {"temperature_c": 24}}],
        memory_reads=1,
        memory_writes=1,
        latency_ms=123.4,
    )
    fetched = rec.get(trace.request_id)
    assert fetched is not None
    assert fetched.final_response == "已调温"
    assert fetched.memory_reads == 1
    assert fetched.latency_ms == 123.4
    assert len(fetched.spans) == 1

    summaries = rec.summaries()
    assert len(summaries) == 1
    assert summaries[0].success is True
    assert summaries[0].tool_call_count == 1


def test_trace_recorder_ring_buffer_eviction() -> None:
    """超出容量时旧 trace 应被淘汰。"""
    rec = TraceRecorder(store_limit=2)
    for i in range(4):
        t = rec.start_trace(session_id="s", user_id="u", model="m", input_text=f"msg{i}")
        rec.end_trace(t, final_response="ok", latency_ms=1.0)
    recent = rec.recent(limit=10)
    assert len(recent) == 2
    assert recent[0].input in ("msg3", "msg2")


# ---------------- Metrics Registry ----------------


def test_metrics_counter_and_histogram() -> None:
    """Counter 与 Histogram 应累计正确。"""
    reg = MetricsRegistry()
    reg.inc_counter("agent_runs_total", labels={"status": "ok"})
    reg.inc_counter("agent_runs_total", labels={"status": "ok"})
    reg.inc_counter("agent_runs_total", labels={"status": "error"})
    snap = reg.snapshot()
    counter_samples = snap["counters"]["agent_runs_total"]
    ok_sample = next(s for s in counter_samples if s["labels"] == {"status": "ok"})
    err_sample = next(s for s in counter_samples if s["labels"] == {"status": "error"})
    assert ok_sample["value"] == 2.0
    assert err_sample["value"] == 1.0

    reg.observe_histogram("agent_run_latency_ms", 50.0, labels={"status": "ok"})
    reg.observe_histogram("agent_run_latency_ms", 500.0, labels={"status": "ok"})
    hist = reg.snapshot()["histograms"]["agent_run_latency_ms"][0]
    assert hist["total"] == 2
    assert hist["sum"] == 550.0


def test_prometheus_exporter_has_counters() -> None:
    """Prometheus 文本应包含计数器。"""
    reg = MetricsRegistry()
    reg.inc_counter("agent_runs_total", labels={"status": "ok"})
    text = render_prometheus(reg)
    assert "# TYPE agent_runs_total counter" in text
    assert 'agent_runs_total{status="ok"} 1' in text


def test_prometheus_exporter_has_histogram() -> None:
    """Prometheus 文本应包含直方图 bucket。"""
    reg = MetricsRegistry()
    reg.observe_histogram("agent_run_latency_ms", 120.0)
    text = render_prometheus(reg)
    assert "# TYPE agent_run_latency_ms histogram" in text
    assert "agent_run_latency_ms_bucket" in text
    assert "agent_run_latency_ms_count" in text


# ---------------- API 端点 ----------------


def test_metrics_endpoint_returns_prometheus_text() -> None:
    """/metrics 应返回 Prometheus 文本。"""
    client = TestClient(app)
    get_metrics_registry().inc_counter("agent_runs_total", labels={"status": "ok"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "agent_runs_total" in resp.text


def test_dashboard_endpoint_returns_snapshot() -> None:
    """/api/v1/observability/dashboard 应返回聚合视图。"""
    client = TestClient(app)
    resp = client.get("/api/v1/observability/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert "recent_traces" in body
    assert "metrics" in body


def test_traces_list_endpoint() -> None:
    """/api/v1/observability/traces 应返回列表。"""
    client = TestClient(app)
    resp = client.get("/api/v1/observability/traces")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_trace_not_found_returns_404() -> None:
    """不存在的 request_id 应返回 404。"""
    client = TestClient(app)
    resp = client.get("/api/v1/observability/traces/does-not-exist")
    assert resp.status_code == 404


def test_trace_detail_after_recording() -> None:
    """记录 trace 后应能通过详情端点查询。"""
    rec = get_trace_recorder()
    trace = rec.start_trace(
        session_id="s-api", user_id="u-api", model="qwen-plus", input_text="hello",
    )
    rec.end_trace(trace, final_response="hi", latency_ms=10.0)
    client = TestClient(app)
    resp = client.get(f"/api/v1/observability/traces/{trace.request_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == trace.request_id
    assert body["final_response"] == "hi"


# ---------------- 二级索引与用户/会话查询 ----------------


def test_trace_recorder_query_by_user() -> None:
    """按 user_id 查询应仅返回该用户的 trace,并按时间倒序。"""
    rec = TraceRecorder(store_limit=10)
    for i in range(3):
        t = rec.start_trace(session_id="s1", user_id="alice", model="m", input_text=f"a{i}")
        rec.end_trace(t, final_response="ok", latency_ms=1.0)
    for i in range(2):
        t = rec.start_trace(session_id="s2", user_id="bob", model="m", input_text=f"b{i}")
        rec.end_trace(t, final_response="ok", latency_ms=1.0)

    alice_traces = rec.by_user("alice", limit=10)
    bob_traces = rec.by_user("bob", limit=10)
    assert len(alice_traces) == 3
    assert len(bob_traces) == 2
    assert all(t.user_id == "alice" for t in alice_traces)
    assert all(t.user_id == "bob" for t in bob_traces)
    assert alice_traces[0].input == "a2"
    assert alice_traces[-1].input == "a0"


def test_trace_recorder_query_by_session() -> None:
    """按 session_id 查询应仅返回该会话内的 trace。"""
    rec = TraceRecorder(store_limit=10)
    for i in range(2):
        t = rec.start_trace(session_id="session-a", user_id="u1", model="m", input_text=f"sa{i}")
        rec.end_trace(t, final_response="ok", latency_ms=1.0)
    t = rec.start_trace(session_id="session-b", user_id="u1", model="m", input_text="sb0")
    rec.end_trace(t, final_response="ok", latency_ms=1.0)

    sess_a = rec.by_session("session-a", limit=10)
    sess_b = rec.by_session("session-b", limit=10)
    assert len(sess_a) == 2
    assert len(sess_b) == 1
    assert sess_a[0].input == "sa1"
    assert sess_b[0].session_id == "session-b"


def test_trace_recorder_query_unknown_user_returns_empty() -> None:
    """未知 user_id / session_id 查询应返回空列表。"""
    rec = TraceRecorder(store_limit=5)
    t = rec.start_trace(session_id="s1", user_id="u1", model="m", input_text="x")
    rec.end_trace(t, final_response="ok")
    assert rec.by_user("ghost") == []
    assert rec.by_session("ghost-session") == []


def test_trace_recorder_ring_buffer_updates_secondary_indexes() -> None:
    """环形缓冲淘汰时 user/session 二级索引同步清理。"""
    rec = TraceRecorder(store_limit=2)
    uid_order = [("u1", "s1"), ("u2", "s2"), ("u3", "s3")]
    for uid, sid in uid_order:
        t = rec.start_trace(session_id=sid, user_id=uid, model="m", input_text=uid)
        rec.end_trace(t, final_response="ok", latency_ms=1.0)

    assert rec.by_user("u1") == []
    assert len(rec.by_user("u2")) == 1
    assert len(rec.by_user("u3")) == 1
    assert rec.by_session("s1") == []
    assert len(rec.by_session("s2")) == 1


def test_trace_recorder_user_query_limit_works() -> None:
    """user/session 查询应支持 limit 参数截断。"""
    rec = TraceRecorder(store_limit=20)
    for i in range(5):
        t = rec.start_trace(session_id="s", user_id="u", model="m", input_text=f"m{i}")
        rec.end_trace(t, final_response="ok", latency_ms=1.0)

    assert len(rec.by_user("u", limit=2)) == 2
    assert len(rec.by_session("s", limit=3)) == 3


# ---------------- 线程安全单例 ----------------


def test_singleton_returns_same_instance() -> None:
    """连续调用应返回同一实例。"""
    a = get_trace_recorder()
    b = get_trace_recorder()
    assert a is b
    c = get_metrics_registry()
    d = get_metrics_registry()
    assert c is d


def test_singleton_thread_safety_concurrent_init() -> None:
    """多线程并发首次调用应只创建一个实例。"""
    import threading
    import importlib
    import app.observability.recorder as rec_mod

    created_instances: list[object] = []
    orig = rec_mod._trace_recorder
    try:
        rec_mod._trace_recorder = None

        def worker() -> None:
            inst = rec_mod.get_trace_recorder()
            created_instances.append(inst)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = created_instances[0]
        assert all(inst is first for inst in created_instances)
        assert len({id(x) for x in created_instances}) == 1
    finally:
        rec_mod._trace_recorder = orig
