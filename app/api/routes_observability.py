# Observability Dashboard 路由 - Trace/Metrics 查询端点(规格第20节,Phase 8)
# 运行指南: 启动 API (python -m app.main) 后访问
#   GET /metrics                                  Prometheus 抓取端点
#   GET /api/v1/observability/dashboard           仪表盘聚合视图
#   GET /api/v1/observability/traces              最近 trace 摘要列表
#   GET /api/v1/observability/traces/{id}         单个 trace 详情
#   GET /api/v1/observability/metrics             指标 JSON 快照

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.observability.prometheus import render_prometheus
from app.observability.recorder import get_metrics_registry, get_trace_recorder

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/dashboard")
async def dashboard(limit: int = Query(20, ge=1, le=200)) -> dict:
    """返回仪表盘聚合视图(最近 trace 摘要 + 指标快照)。"""
    traces = get_trace_recorder().summaries(limit=limit)
    metrics = get_metrics_registry().snapshot()
    return {
        "recent_traces": [t.model_dump() for t in traces],
        "metrics": metrics,
        "trace_count": len(traces),
    }


@router.get("/traces")
async def list_traces(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    """返回最近 N 条 trace 的摘要。"""
    summaries = get_trace_recorder().summaries(limit=limit)
    return [s.model_dump() for s in summaries]


@router.get("/traces/{request_id}")
async def get_trace(request_id: str) -> dict:
    """返回指定 request_id 的 trace 详情。"""
    trace = get_trace_recorder().get(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace {request_id} not found")
    return trace.model_dump()


@router.get("/metrics")
async def metrics_snapshot() -> dict:
    """返回指标 JSON 快照(供调试/Dashboard)。"""
    return get_metrics_registry().snapshot()


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """返回 Prometheus exposition 文本(被 /metrics 根端点转发)。"""
    return render_prometheus(get_metrics_registry())


@router.get("/health", response_class=PlainTextResponse)
async def observability_health() -> str:
    """可观测子系统健康检查。"""
    return "ok"
