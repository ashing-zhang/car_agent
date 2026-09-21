# Bad Case 评估脚本 - 仅对已识别的 Bad Case 进行重新评估(规格第19节扩展)
# 运行指南:
#   python -m scripts.run_bad_case_eval
#   可选参数(通过环境变量或直接修改 BadCaseEvalConfig):
#     BAD_CASE_SOURCE:  bad case 来源, "summary_json" 或 "individual_files" (默认 summary_json)
#     BAD_CASE_JSON:    bad case 汇总 JSON 路径 (默认 reports/bad_cases_from_index.json)
#     BAD_CASE_DIR:     独立 bad case JSON 文件目录 (默认 reports/bad_cases_from_index)
#   输出: reports/bad_case_re_eval/ 目录下的 trace JSON、metrics 报告、汇总 CSV
#
# 设计说明(开闭原则 + 配置驱动):
#   - 通过 BadCaseEvalConfig 数据类驱动所有可配置项,无需修改代码即可切换数据源/输出路径
#   - BadCaseLoader 抽象了两种数据源(汇总 JSON / 独立文件),新增来源时只需扩展 loader 逻辑
#   - 报告写入逻辑复用 run_eval.py 中的 CaseResult 序列化格式,与全量评估保持一致

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.dataset import EvalCase, load_scenarios
from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics import CaseResult, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BadCaseEvalConfig:
    """Bad Case 重评估运行配置(配置驱动,开闭原则:新增配置不修改运行逻辑)。"""

    bad_case_source: str = "summary_json"
    bad_case_summary_json: Path = PROJECT_ROOT / "reports" / "bad_cases_from_index.json"
    bad_case_individual_dir: Path = PROJECT_ROOT / "reports" / "bad_cases_from_index"
    scenarios_dir: Path = PROJECT_ROOT / "app" / "evaluation" / "scenarios"
    output_dir: Path = PROJECT_ROOT / "reports" / "bad_case_re_eval"
    traces_subdir: str = "traces"
    compute_full_metrics: bool = True
    case_id_filter: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "BadCaseEvalConfig":
        """从环境变量构建配置(便于 CI/脚本化调用)。"""
        cfg = cls()
        if src := os.environ.get("BAD_CASE_SOURCE"):
            cfg.bad_case_source = src
        if json_path := os.environ.get("BAD_CASE_JSON"):
            cfg.bad_case_summary_json = Path(json_path)
        if dir_path := os.environ.get("BAD_CASE_DIR"):
            cfg.bad_case_individual_dir = Path(dir_path)
        if out_dir := os.environ.get("BAD_CASE_OUTPUT_DIR"):
            cfg.output_dir = Path(out_dir)
        if filter_str := os.environ.get("BAD_CASE_ID_FILTER"):
            cfg.case_id_filter = [s.strip() for s in filter_str.split(",") if s.strip()]
        return cfg


class BadCaseLoader:
    """Bad Case 加载器(开闭原则:新增数据源时可扩展此类)。"""

    def __init__(self, config: BadCaseEvalConfig) -> None:
        """初始化加载器,持有配置引用。"""
        self._cfg = config

    def load_case_ids(self) -> list[str]:
        """从配置的数据源加载 bad case 的 case_id 列表,去重并保持顺序。"""
        source = self._cfg.bad_case_source.lower()
        case_ids: list[str] = []
        seen: set[str] = set()

        if source == "summary_json":
            ids = self._load_from_summary_json()
        elif source in ("individual_files", "individual", "files"):
            ids = self._load_from_individual_files()
        else:
            logger.warning(
                "Unknown bad_case_source=%s, fallback to summary_json", self._cfg.bad_case_source,
            )
            ids = self._load_from_summary_json()

        for cid in ids:
            if cid in seen:
                continue
            seen.add(cid)
            if self._cfg.case_id_filter and cid not in set(self._cfg.case_id_filter):
                continue
            case_ids.append(cid)

        logger.info("Loaded %d unique bad case ids from source=%s", len(case_ids), source)
        return case_ids

    def _load_from_summary_json(self) -> list[str]:
        """从汇总 JSON (reports/bad_cases_from_index.json) 提取 case_id。"""
        json_path = self._cfg.bad_case_summary_json
        if not json_path.exists():
            logger.error("Bad case summary JSON not found: %s", json_path)
            logger.info(
                "Hint: run `python -m reports.extract_bad_cases` first to generate the summary, "
                "or set BAD_CASE_SOURCE=individual_files to read per-case JSONs directly.",
            )
            return []
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse bad case summary JSON: %s", exc)
            return []
        bad_cases = data.get("bad_cases", []) if isinstance(data, dict) else []
        return [entry.get("case_id", "") for entry in bad_cases if isinstance(entry, dict) and entry.get("case_id")]

    def _load_from_individual_files(self) -> list[str]:
        """从 bad_cases_from_index/ 目录下的独立 JSON 文件提取 case_id。"""
        in_dir = self._cfg.bad_case_individual_dir
        if not in_dir.exists():
            logger.error("Bad case individual dir not found: %s", in_dir)
            return []
        ids: list[str] = []
        for fp in sorted(in_dir.glob("*_bad_case.json")):
            stem = fp.stem
            suffix = "_bad_case"
            if stem.endswith(suffix):
                case_id = stem[: -len(suffix)]
            else:
                case_id = stem
            ids.append(case_id)
        return ids


def _load_target_cases(case_ids: list[str], config: BadCaseEvalConfig) -> list[EvalCase]:
    """根据 case_id 列表从 scenarios 目录加载对应 EvalCase。"""
    all_scenarios = load_scenarios(config.scenarios_dir, auto_generate=False, auto_persist=False)
    id_to_case: dict[str, EvalCase] = {c.id: c for c in all_scenarios}
    missing: list[str] = []
    target: list[EvalCase] = []
    for cid in case_ids:
        case = id_to_case.get(cid)
        if case is None:
            missing.append(cid)
            continue
        target.append(case)
    if missing:
        logger.warning("Skipped %d bad cases not found in scenarios dir: %s", len(missing), missing)
    logger.info("Resolved %d/%d bad cases to EvalCase objects", len(target), len(case_ids))
    return target


def _write_case_trace(result: CaseResult, traces_dir: Path) -> Path:
    """为单个 case 写入独立 trace JSON,返回写入路径。"""
    traces_dir.mkdir(parents=True, exist_ok=True)
    case_id = result.case.id
    trace_file = traces_dir / f"{case_id}_trace.json"
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
        "intent_correct": result.intent_correct,
        "trace": result.trace.model_dump(mode="json") if hasattr(result.trace, "model_dump") else result.trace,
    }
    trace_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace_file


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
    if result.case.expected_reject and not reasons:
        if actual & {"steering", "brake", "throttle"}:
            pass
        else:
            reasons.append("REJECTED_AS_EXPECTED")
    if not reasons:
        reasons.append("UNKNOWN_FAILURE")
    return "; ".join(reasons)


def _write_results_summary(results: list[CaseResult], output_dir: Path) -> tuple[Path, Path]:
    """写入 CSV 逐 case 摘要 + 修复对比 JSON,返回 (csv_path, compare_json_path)。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bad_case_re_eval_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "category", "user_input", "expected_tools", "actual_tools",
            "success_prev", "success_now", "fixed", "intent_correct",
            "hallucinated_tools", "has_exception", "failure_reason", "latency_ms",
        ])
        for r in results:
            writer.writerow([
                r.case.id,
                r.case.category,
                r.case.user,
                "|".join(r.case.expected_tools),
                "|".join(r.actual_tools),
                False,
                r.success,
                r.success,
                r.intent_correct,
                "|".join(r.hallucinated_tools),
                r.trace.error_traceback is not None,
                _get_failure_reason(r) if not r.success else "PASSED",
                f"{r.latency_ms:.2f}",
            ])

    fixed_ids = [r.case.id for r in results if r.success]
    still_bad = [r.case.id for r in results if not r.success]
    compare = {
        "total_bad_cases_re_eval": len(results),
        "fixed_count": len(fixed_ids),
        "still_bad_count": len(still_bad),
        "fix_rate": (len(fixed_ids) / len(results)) if results else 0.0,
        "fixed_case_ids": fixed_ids,
        "still_bad_case_ids": still_bad,
        "per_case": [
            {
                "case_id": r.case.id,
                "category": r.case.category,
                "success": r.success,
                "actual_tools": r.actual_tools,
                "expected_tools": r.case.expected_tools,
                "failure_reason": "" if r.success else _get_failure_reason(r),
            }
            for r in results
        ],
    }
    compare_path = output_dir / "bad_case_fix_report.json"
    compare_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, compare_path


def run_bad_case_evaluation(config: BadCaseEvalConfig | None = None) -> list[CaseResult]:
    """执行 Bad Case 重评估主流程,返回结果列表供调用方进一步分析。

    Args:
        config: 运行时配置,None 时从环境变量 + 默认值构造

    Returns:
        按 case_id 输入顺序排列的 CaseResult 列表(即使失败也会携带 trace 错误信息)
    """
    cfg = config or BadCaseEvalConfig.from_env()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = cfg.output_dir / cfg.traces_subdir

    loader = BadCaseLoader(cfg)
    case_ids = loader.load_case_ids()
    if not case_ids:
        logger.warning("No bad case ids loaded, exiting without evaluation")
        return []

    target_cases = _load_target_cases(case_ids, cfg)
    if not target_cases:
        logger.warning("No target EvalCase resolved, exiting without evaluation")
        return []

    evaluator = Evaluator()
    logger.info("Starting bad case re-evaluation for %d cases...", len(target_cases))
    results = evaluator.run(target_cases)

    for r in results:
        _write_case_trace(r, traces_dir)
    logger.info("Traces written to %s", traces_dir)

    csv_path, compare_path = _write_results_summary(results, cfg.output_dir)
    logger.info("Bad case summary CSV written to %s", csv_path)
    logger.info("Bad case fix report JSON written to %s", compare_path)

    if cfg.compute_full_metrics:
        report = compute_metrics(results)
        metrics_path = cfg.output_dir / "bad_case_metrics.json"
        metrics_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Metrics report written to %s", metrics_path)

    fixed = sum(1 for r in results if r.success)
    errored = sum(1 for r in results if r.trace.error_traceback)
    print("\n=== Bad Case Re-Evaluation Report ===")
    print(f"Re-evaluated cases: {len(results)}")
    print(f"Fixed (now pass):   {fixed} / {len(results)}  ({(fixed / len(results)):.0%})" if results else "Fixed (now pass):   N/A")
    print(f"Still failing:      {len(results) - fixed} / {len(results)}")
    print(f"Errored during eval:{errored} / {len(results)}")
    print(f"Output directory:   {cfg.output_dir}")
    print(f"Traces:             {traces_dir}")
    print(f"Fix report:         {compare_path}")
    print(f"Summary CSV:        {csv_path}")

    return results


def main() -> None:
    """CLI 入口:读取环境变量并执行 Bad Case 重评估。"""
    run_bad_case_evaluation()


if __name__ == "__main__":
    main()
