# Evaluation 数据集持久化单元测试 - 验证场景文件保存/加载(规格第18节)
# 运行指南: pytest tests/unit/test_dataset_persistence.py -v

from pathlib import Path

from app.evaluation.dataset import EvalCase, generate_scenarios, load_scenarios, save_scenarios


def test_save_scenarios_writes_per_case_json(tmp_path: Path) -> None:
    """验证 save_scenarios 按 category 分目录、每 case 写一个 JSON。"""
    cases = generate_scenarios()
    root = save_scenarios(cases, tmp_path)
    assert root == tmp_path
    for cat in {"vehicle_query", "vehicle_control", "navigation", "media",
                "memory", "multi_step", "multimodal", "ambiguous", "unsafe_request"}:
        assert (tmp_path / cat).is_dir(), f"missing category dir: {cat}"
    json_count = len(list(tmp_path.rglob("*.json")))
    assert json_count == len(cases)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """验证保存后再加载,内容与原始列表一致(按 id 排序后比较)。"""
    original = generate_scenarios()
    save_scenarios(original, tmp_path)
    loaded = load_scenarios(tmp_path, auto_generate=False, auto_persist=False)
    assert len(loaded) == len(original)
    by_id_orig = {c.id: c for c in original}
    by_id_load = {c.id: c for c in loaded}
    assert set(by_id_orig.keys()) == set(by_id_load.keys())
    for cid in by_id_orig:
        assert by_id_load[cid].model_dump() == by_id_orig[cid].model_dump()


def test_load_scenarios_auto_generate_and_persist(tmp_path: Path) -> None:
    """验证空目录下 load_scenarios 自动生成并固化,第二次加载走文件路径。"""
    assert not any(tmp_path.rglob("*.json"))
    first = load_scenarios(tmp_path, auto_generate=True, auto_persist=True)
    json_files = list(tmp_path.rglob("*.json"))
    assert len(json_files) == len(first)
    second = load_scenarios(tmp_path, auto_generate=False, auto_persist=False)
    assert len(second) == len(first)
    assert {c.id for c in second} == {c.id for c in first}


def test_load_scenarios_invalid_file_skipped(tmp_path: Path) -> None:
    """验证某个 JSON 文件损坏时被跳过,其余正常加载。"""
    cases = generate_scenarios()[:5]
    save_scenarios(cases, tmp_path)
    bad_file = tmp_path / cases[0].category / f"{cases[0].id}.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    loaded = load_scenarios(tmp_path, auto_generate=False, auto_persist=False)
    assert len(loaded) == len(cases) - 1
    assert cases[0].id not in {c.id for c in loaded}


def test_load_scenarios_no_data_no_generate_raises(tmp_path: Path) -> None:
    """验证空目录 + auto_generate=False 时抛出 FileNotFoundError。"""
    try:
        load_scenarios(tmp_path, auto_generate=False, auto_persist=False)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_eval_case_id_filename_matches_report(tmp_path: Path) -> None:
    """验证 case_id 与文件名一致,便于 eval report 对应追踪。"""
    case = EvalCase(
        id="trace_demo_001",
        user="测试对应",
        expected_tools=["get_vehicle_status"],
        category="vehicle_query",
    )
    save_scenarios([case], tmp_path)
    expected_path = tmp_path / "vehicle_query" / "trace_demo_001.json"
    assert expected_path.exists()
    roundtrip = EvalCase.model_validate_json(expected_path.read_text(encoding="utf-8"))
    assert roundtrip.id == "trace_demo_001"
    assert roundtrip.user == "测试对应"
