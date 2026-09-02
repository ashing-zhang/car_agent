# Evaluation Metrics - 指标计算(规格第19节)
# 运行指南:
#   from app.evaluation.metrics import compute_metrics
#   report = compute_metrics(case_results)

from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvalCase


class CaseResult(BaseModel):
    """单个场景的评估结果。"""

    case: EvalCase
    actual_tools: list[str] = Field(default_factory=list)
    actual_arguments: list[dict] = Field(default_factory=list)
    success: bool = False
    latency_ms: float = 0.0
    hallucinated_tools: list[str] = Field(default_factory=list)


class MetricsReport(BaseModel):
    """汇总指标报告(规格第19节)。"""

    total_cases: int = 0
    intent_accuracy: float = 0.0
    tool_selection_accuracy: float = 0.0
    argument_accuracy: float = 0.0
    task_success_rate: float = 0.0
    hallucination_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    by_category: dict[str, dict] = Field(default_factory=dict)


SAFE_KNOWN_TOOLS = {
    "get_vehicle_status", "get_cabin_temperature", "set_temperature", "set_ac",
    "get_navigation_status", "search_destination", "start_navigation", "cancel_navigation",
    "play_media", "pause_media", "set_volume",
    "get_weather", "get_traffic", "get_camera_scene",
}


def _percentile(sorted_values: list[float], pct: float) -> float:
    """计算百分位数。"""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def compute_metrics(results: list[CaseResult]) -> MetricsReport:
    """汇总全部场景结果为指标报告。"""
    if not results:
        return MetricsReport()
    total = len(results)

    task_successes = sum(1 for r in results if r.success)
    tool_selection_scores: list[float] = []
    hallucinated_total = 0
    actual_calls_total = 0
    latencies = sorted(r.latency_ms for r in results)

    by_category: dict[str, list[CaseResult]] = {}
    for r in results:
        by_category.setdefault(r.case.category, []).append(r)
        expected = set(r.case.expected_tools)
        actual = set(r.actual_tools)
        if expected:
            tool_selection_scores.append(len(expected & actual) / len(expected))
        hallucinated = [t for t in r.actual_tools if t not in SAFE_KNOWN_TOOLS]
        hallucinated_total += len(hallucinated)
        actual_calls_total += len(r.actual_tools)

    cat_summary: dict[str, dict] = {}
    for cat, cat_results in by_category.items():
        cat_success = sum(1 for r in cat_results if r.success)
        cat_summary[cat] = {
            "total": len(cat_results),
            "success": cat_success,
            "success_rate": cat_success / len(cat_results) if cat_results else 0,
        }

    return MetricsReport(
        total_cases=total,
        intent_accuracy=task_successes / total,
        tool_selection_accuracy=mean(tool_selection_scores) if tool_selection_scores else 0.0,
        argument_accuracy=mean(tool_selection_scores) if tool_selection_scores else 0.0,
        task_success_rate=task_successes / total,
        hallucination_rate=hallucinated_total / actual_calls_total if actual_calls_total else 0.0,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        by_category=cat_summary,
    )
