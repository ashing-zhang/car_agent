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
    print("\nBy category:")
    for cat, stats in report.by_category.items():
        print(f"  {cat}: {stats['success']}/{stats['total']} ({stats['success_rate']:.0%})")
    print(f"\nOpen traces index to debug bad cases: {index_path}")


if __name__ == "__main__":
    main()
