# 评估运行脚本 - 运行 Automotive Agent Benchmark 并输出 JSON/CSV 报告(规格第19节)
# 运行指南:
#   python -m scripts.run_eval
#   输出: reports/eval_report.json, reports/eval_summary.csv, reports/traces/{case_id}_trace.json

import csv
import json
import logging
from pathlib import Path

from app.evaluation.dataset import load_scenarios
from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics import CaseResult, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
TRACES_DIR = REPORTS_DIR / "traces"
BAD_CASES_DIR = REPORTS_DIR / "bad_cases"


def _write_case_trace(result: CaseResult) -> Path:
    """为单个 case 写入独立的 trace JSON 文件,返回文件路径。"""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    case_id = result.case.id
    trace_file = TRACES_DIR / f"{case_id}_trace.json"

    payload = {
        "case_id": case_id,
        "category": result.case.category,
        "user_input": result.case.user,
        "expected_tools": result.case.expected_tools,
        "expected_reject": result.case.expected_reject,
        "success": result.success,
        "latency_ms": round(result.latency_ms, 2),
        "actual_tools": result.actual_tools,
        "actual_arguments": result.actual_arguments,
        "hallucinated_tools": result.hallucinated_tools,
        "trace": result.trace.model_dump(mode="json") if hasattr(result.trace, "model_dump") else result.trace,
    }
    trace_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace_file


def _write_traces_index(results: list[CaseResult]) -> Path:
    """写入 traces 索引文件,便于批量定位 bad case 的 trace 文件。"""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for r in results:
        index.append({
            "case_id": r.case.id,
            "category": r.case.category,
            "success": r.success,
            "latency_ms": round(r.latency_ms, 2),
            "expected_tools": r.case.expected_tools,
            "actual_tools": r.actual_tools,
            "has_error": r.trace.error_traceback is not None,
            "trace_file": f"traces/{r.case.id}_trace.json",
        })
    index_path = REPORTS_DIR / "traces_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path


def _get_failure_reason(result: CaseResult) -> str:
    """分析单个 case 的失败原因并返回可读描述。"""
    if result.trace.error_traceback:
        return f"EXCEPTION: {result.trace.error_message or 'Unknown error'}"
    expected = set(result.case.expected_tools)
    actual = set(result.actual_tools)
    missing = expected - actual
    unexpected = actual - expected
    reasons: list[str] = []
    if result.case.expected_reject and (actual & {"steering", "brake", "throttle"}):
        reasons.append("SAFETY_VIOLATION: called safety-critical tools when rejection expected")
    if missing:
        reasons.append(f"MISSING_TOOLS: {sorted(missing)}")
    if result.hallucinated_tools:
        reasons.append(f"HALLUCINATED: {result.hallucinated_tools}")
    if unexpected and not result.case.expected_reject:
        known_unexpected = unexpected - set(result.hallucinated_tools)
        if known_unexpected:
            reasons.append(f"UNEXPECTED_TOOLS: {sorted(known_unexpected)}")
    if not reasons:
        reasons.append("UNKNOWN_FAILURE")
    return "; ".join(reasons)


def _write_bad_cases_json(results: list[CaseResult]) -> Path:
    """将 success=false 的 case 详细信息写入 bad_cases 汇总 JSON 文件。"""
    BAD_CASES_DIR.mkdir(parents=True, exist_ok=True)
    bad_cases = [r for r in results if not r.success]
    payload = {
        "total_bad_cases": len(bad_cases),
        "total_cases": len(results),
        "bad_case_rate": (len(bad_cases) / len(results)) if results else 0.0,
        "bad_cases": [],
    }
    for r in bad_cases:
        failure_reason = _get_failure_reason(r)
        case_entry = {
            "case_id": r.case.id,
            "category": r.case.category,
            "user_input": r.case.user,
            "expected_tools": r.case.expected_tools,
            "expected_reject": r.case.expected_reject,
            "actual_tools": r.actual_tools,
            "actual_arguments": r.actual_arguments,
            "hallucinated_tools": r.hallucinated_tools,
            "success": r.success,
            "latency_ms": round(r.latency_ms, 2),
            "intent_correct": r.intent_correct,
            "intent_classified_type": r.trace.intent_classified_type,
            "intent_ground_truth": r.trace.intent_ground_truth,
            "failure_reason": failure_reason,
            "has_exception": r.trace.error_traceback is not None,
            "error_message": r.trace.error_message,
            "final_response": r.trace.final_response,
            "plan_steps": r.trace.plan_steps,
            "trace_file": f"../traces/{r.case.id}_trace.json",
        }
        payload["bad_cases"].append(case_entry)
        bad_case_file = BAD_CASES_DIR / f"{r.case.id}_bad_case.json"
        bad_case_file.write_text(json.dumps(case_entry, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = BAD_CASES_DIR / "bad_cases_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _write_bad_cases_csv(results: list[CaseResult]) -> Path:
    """将 success=false 的 case 摘要信息写入 CSV 文件,便于快速浏览。"""
    BAD_CASES_DIR.mkdir(parents=True, exist_ok=True)
    bad_cases = [r for r in results if not r.success]
    csv_path = BAD_CASES_DIR / "bad_cases_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "category", "user_input", "expected_tools", "actual_tools",
            "hallucinated_tools", "intent_correct", "intent_gt_vs_pred",
            "has_exception", "failure_reason", "latency_ms", "trace_file",
        ])
        for r in bad_cases:
            intent_vs = f"{r.trace.intent_ground_truth}->{r.trace.intent_classified_type}"
            writer.writerow([
                r.case.id,
                r.case.category,
                r.case.user,
                "|".join(r.case.expected_tools),
                "|".join(r.actual_tools),
                "|".join(r.hallucinated_tools),
                r.intent_correct,
                intent_vs,
                r.trace.error_traceback is not None,
                _get_failure_reason(r),
                f"{r.latency_ms:.2f}",
                f"../traces/{r.case.id}_trace.json",
            ])
    return csv_path


def main() -> None:
    """运行评估并生成报告文件。"""
    scenarios = load_scenarios()
    logger.info("Loaded %d evaluation scenarios", len(scenarios))

    evaluator = Evaluator()
    results = evaluator.run(scenarios)
    report = compute_metrics(results)

    REPORTS_DIR.mkdir(exist_ok=True)
    json_path = REPORTS_DIR / "eval_report.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("JSON report written to %s", json_path)

    csv_path = REPORTS_DIR / "eval_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "category", "user", "expected_tools", "actual_tools",
            "success", "latency_ms", "hallucinated_tools", "trace_file",
        ])
        for r in results:
            writer.writerow([
                r.case.id, r.case.category, r.case.user,
                "|".join(r.case.expected_tools), "|".join(r.actual_tools),
                r.success, f"{r.latency_ms:.2f}",
                "|".join(r.hallucinated_tools),
                f"traces/{r.case.id}_trace.json",
            ])
    logger.info("CSV report written to %s", csv_path)

    for r in results:
        _write_case_trace(r)
    logger.info("Trace files written to %s", TRACES_DIR)

    index_path = _write_traces_index(results)
    logger.info("Traces index written to %s", index_path)

    bad_cases_json_path = _write_bad_cases_json(results)
    logger.info("Bad cases summary JSON written to %s", bad_cases_json_path)

    bad_cases_csv_path = _write_bad_cases_csv(results)
    logger.info("Bad cases summary CSV written to %s", bad_cases_csv_path)

    failed = sum(1 for r in results if not r.success)
    errored = sum(1 for r in results if r.trace.error_traceback)
    print("\n=== AutoAgent Evaluation Report ===")
    print(f"Total cases:        {report.total_cases}")
    print(f"Task success rate:  {report.task_success_rate:.1%}")
    print(f"Tool selection acc:  {report.tool_selection_accuracy:.1%}")
    print(f"Argument accuracy:  {report.argument_accuracy:.1%}")
    print(f"Intent accuracy:    {report.intent_accuracy:.1%}")
    print(f"Hallucination rate: {report.hallucination_rate:.1%}")
    print(f"Latency P50/P95/P99: {report.latency_p50_ms:.1f}/{report.latency_p95_ms:.1f}/{report.latency_p99_ms:.1f} ms")
    print(f"Failed cases:       {failed} / {report.total_cases}")
    print(f"Errored cases:      {errored} / {report.total_cases}")
    print(f"Trace directory:    {TRACES_DIR}")
    print(f"Bad cases dir:      {BAD_CASES_DIR}")
    print("\nBy category:")
    for cat, stats in report.by_category.items():
        print(f"  {cat}: {stats['success']}/{stats['total']} ({stats['success_rate']:.0%})")
    print(f"\nOpen traces index to debug bad cases: {index_path}")
    if failed > 0:
        print(f"Bad cases summary JSON: {bad_cases_json_path}")
        print(f"Bad cases summary CSV:  {bad_cases_csv_path}")


if __name__ == "__main__":
    main()
