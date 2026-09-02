# 评估运行脚本 - 运行 Automotive Agent Benchmark 并输出 JSON/CSV 报告(规格第19节)
# 运行指南:
#   python -m scripts.run_eval
#   输出: reports/eval_report.json, reports/eval_summary.csv

import csv
import json
import logging
from pathlib import Path

from app.evaluation.dataset import load_scenarios
from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


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
        writer.writerow(["id", "category", "user", "expected_tools", "actual_tools", "success", "latency_ms"])
        for r in results:
            writer.writerow([
                r.case.id, r.case.category, r.case.user,
                "|".join(r.case.expected_tools), "|".join(r.actual_tools),
                r.success, f"{r.latency_ms:.2f}",
            ])
    logger.info("CSV report written to %s", csv_path)

    print("\n=== AutoAgent Evaluation Report ===")
    print(f"Total cases:        {report.total_cases}")
    print(f"Task success rate:  {report.task_success_rate:.1%}")
    print(f"Tool selection acc:  {report.tool_selection_accuracy:.1%}")
    print(f"Argument accuracy:  {report.argument_accuracy:.1%}")
    print(f"Intent accuracy:    {report.intent_accuracy:.1%}")
    print(f"Hallucination rate: {report.hallucination_rate:.1%}")
    print(f"Latency P50/P95/P99: {report.latency_p50_ms:.1f}/{report.latency_p95_ms:.1f}/{report.latency_p99_ms:.1f} ms")
    print("\nBy category:")
    for cat, stats in report.by_category.items():
        print(f"  {cat}: {stats['success']}/{stats['total']} ({stats['success_rate']:.0%})")


if __name__ == "__main__":
    main()
