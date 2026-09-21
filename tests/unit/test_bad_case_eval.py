# Bad Case 重评估脚本单元测试 - 验证配置/加载/报告写入逻辑(不实际调用 LLM)
# 运行指南:
#   pytest tests/unit/test_bad_case_eval.py -v
#   注意: 所有测试均使用临时目录 + mock 数据,不触发真实 LLM 调用

import json
import tempfile
from pathlib import Path

from scripts.run_bad_case_eval import (
    BadCaseEvalConfig,
    BadCaseLoader,
    _get_failure_reason,
    _load_target_cases,
    _write_results_summary,
)


def test_bad_case_eval_config_defaults() -> None:
    """验证默认配置值与项目结构对齐。"""
    cfg = BadCaseEvalConfig()
    assert cfg.bad_case_source == "summary_json"
    assert cfg.bad_case_summary_json.name == "bad_cases_from_index.json"
    assert cfg.bad_case_individual_dir.name == "bad_cases_from_index"
    assert cfg.output_dir.name == "bad_case_re_eval"
    assert cfg.traces_subdir == "traces"
    assert cfg.compute_full_metrics is True
    assert cfg.case_id_filter == []


def test_bad_case_loader_from_summary_json() -> None:
    """验证 BadCaseLoader 从汇总 JSON 正确提取 case_id 并去重。"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        summary_json = td_path / "bad_cases_summary.json"
        payload = {
            "bad_cases": [
                {"case_id": "mem_003", "category": "memory"},
                {"case_id": "ms_001", "category": "multi_step"},
                {"case_id": "mem_003", "category": "memory"},
                {"case_id": "mm_001", "category": "multimodal"},
            ],
        }
        summary_json.write_text(json.dumps(payload), encoding="utf-8")

        cfg = BadCaseEvalConfig(
            bad_case_source="summary_json",
            bad_case_summary_json=summary_json,
            bad_case_individual_dir=td_path / "x",
            scenarios_dir=td_path,
            output_dir=td_path / "out",
        )
        loader = BadCaseLoader(cfg)
        ids = loader.load_case_ids()
        assert ids == ["mem_003", "ms_001", "mm_001"]


def test_bad_case_loader_from_summary_json_missing() -> None:
    """验证汇总 JSON 不存在时加载器返回空列表并记录日志(不抛异常)。"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cfg = BadCaseEvalConfig(
            bad_case_source="summary_json",
            bad_case_summary_json=td_path / "not_exist.json",
            bad_case_individual_dir=td_path,
            scenarios_dir=td_path,
            output_dir=td_path / "out",
        )
        loader = BadCaseLoader(cfg)
        ids = loader.load_case_ids()
        assert ids == []


def test_bad_case_loader_from_individual_files() -> None:
    """验证 BadCaseLoader 从独立 bad case 文件目录扫描并提取 case_id。"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        bad_dir = td_path / "bad_cases"
        bad_dir.mkdir()
        for cid in ["mem_003", "ms_001", "mm_001"]:
            (bad_dir / f"{cid}_bad_case.json").write_text("{}", encoding="utf-8")

        cfg = BadCaseEvalConfig(
            bad_case_source="individual_files",
            bad_case_summary_json=td_path / "x",
            bad_case_individual_dir=bad_dir,
            scenarios_dir=td_path,
            output_dir=td_path / "out",
        )
        loader = BadCaseLoader(cfg)
        ids = loader.load_case_ids()
        assert sorted(ids) == ["mem_003", "mm_001", "ms_001"]


def test_bad_case_loader_case_id_filter() -> None:
    """验证 case_id_filter 可按用户指定范围过滤 bad case。"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        summary_json = td_path / "sum.json"
        payload = {
            "bad_cases": [
                {"case_id": "mem_003"},
                {"case_id": "ms_001"},
                {"case_id": "mm_001"},
            ],
        }
        summary_json.write_text(json.dumps(payload), encoding="utf-8")
        cfg = BadCaseEvalConfig(
            bad_case_source="summary_json",
            bad_case_summary_json=summary_json,
            bad_case_individual_dir=td_path,
            scenarios_dir=td_path,
            output_dir=td_path / "out",
            case_id_filter=["ms_001", "mm_001"],
        )
        loader = BadCaseLoader(cfg)
        ids = loader.load_case_ids()
        assert ids == ["ms_001", "mm_001"]


def test_load_target_cases_resolves_and_skips_missing(tmp_path: Path) -> None:
    """验证 _load_target_cases 从 scenarios 目录解析对应 case,缺失时记录并跳过。"""
    scenarios_dir = tmp_path / "scenarios"
    mem_dir = scenarios_dir / "memory"
    ms_dir = scenarios_dir / "multi_step"
    mem_dir.mkdir(parents=True)
    ms_dir.mkdir(parents=True)

    mem_case = {"id": "mem_003", "user": "有点冷", "expected_tools": ["set_temperature"],
                "expected_success": True, "category": "memory", "expected_reject": False}
    ms_case = {"id": "ms_001", "user": "去公司顺便放音乐",
               "expected_tools": ["search_destination", "start_navigation", "play_media"],
               "expected_success": True, "category": "multi_step", "expected_reject": False}
    (mem_dir / "mem_003.json").write_text(json.dumps(mem_case), encoding="utf-8")
    (ms_dir / "ms_001.json").write_text(json.dumps(ms_case), encoding="utf-8")

    cfg = BadCaseEvalConfig(
        scenarios_dir=scenarios_dir,
        bad_case_summary_json=tmp_path / "x",
        bad_case_individual_dir=tmp_path / "y",
        output_dir=tmp_path / "out",
    )

    resolved = _load_target_cases(["mem_003", "ms_001", "ghost_999"], cfg)
    assert [c.id for c in resolved] == ["mem_003", "ms_001"]
    assert resolved[0].category == "memory"
    assert resolved[1].expected_tools == ["search_destination", "start_navigation", "play_media"]


def test_get_failure_reason_missing_and_unexpected() -> None:
    """验证 _get_failure_reason 正确组合 MISSING/UNEXPECTED 原因。"""
    from app.evaluation.dataset import EvalCase
    from app.evaluation.metrics import CaseResult, CaseTrace

    case = EvalCase(id="t", user="u", expected_tools=["set_temperature"], category="memory")
    result = CaseResult(
        case=case,
        actual_tools=["get_cabin_temperature", "get_vehicle_status"],
        success=False,
        trace=CaseTrace(),
    )
    reason = _get_failure_reason(result)
    assert "MISSING_TOOLS:" in reason
    assert "set_temperature" in reason
    assert "UNEXPECTED_TOOLS:" in reason
    assert "get_cabin_temperature" in reason or "get_vehicle_status" in reason


def test_get_failure_reason_exception() -> None:
    """验证异常场景优先使用 EXCEPTION 前缀。"""
    from app.evaluation.dataset import EvalCase
    from app.evaluation.metrics import CaseResult, CaseTrace

    case = EvalCase(id="t", user="u", expected_tools=["set_temperature"], category="memory")
    result = CaseResult(
        case=case,
        actual_tools=[],
        success=False,
        trace=CaseTrace(error_traceback="traceback-text", error_message="boom"),
    )
    assert _get_failure_reason(result).startswith("EXCEPTION:")


def test_write_results_summary_generates_csv_and_json(tmp_path: Path) -> None:
    """验证 _write_results_summary 产出 CSV 与修复报告 JSON,且 fixed/still_bad 计数正确。"""
    from app.evaluation.dataset import EvalCase
    from app.evaluation.metrics import CaseResult, CaseTrace

    case_ok = EvalCase(id="mem_003", user="有点冷", expected_tools=["set_temperature"], category="memory")
    case_fail = EvalCase(id="ms_001", user="去公司顺便放音乐",
                         expected_tools=["search_destination", "start_navigation", "play_media"],
                         category="multi_step")
    results = [
        CaseResult(case=case_ok, actual_tools=["set_temperature"], success=True, latency_ms=1000.0, trace=CaseTrace()),
        CaseResult(case=case_fail, actual_tools=["play_media"], success=False, latency_ms=1200.0, trace=CaseTrace()),
    ]

    csv_path, compare_path = _write_results_summary(results, tmp_path)

    assert csv_path.exists()
    assert compare_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "mem_003" in csv_text
    assert "ms_001" in csv_text
    assert "PASSED" in csv_text
    assert "MISSING_TOOLS:" in csv_text

    compare = json.loads(compare_path.read_text(encoding="utf-8"))
    assert compare["total_bad_cases_re_eval"] == 2
    assert compare["fixed_count"] == 1
    assert compare["still_bad_count"] == 1
    assert compare["fix_rate"] == 0.5
    assert compare["fixed_case_ids"] == ["mem_003"]
    assert compare["still_bad_case_ids"] == ["ms_001"]
    assert compare["per_case"][0]["success"] is True
    assert compare["per_case"][1]["success"] is False
