# Evaluation Trace 记录单元测试 - 验证每个 case 的完整数据流记录
# 运行指南: pytest tests/unit/test_evaluation_trace.py -v

from __future__ import annotations

from pathlib import Path

from app.evaluation.dataset import EvalCase
from app.evaluation.evaluator import Evaluator, _message_to_record
from app.evaluation.metrics import (
    CaseResult,
    CaseTrace,
    MessageRecord,
    NodeExecutionRecord,
    ToolExecutionRecord,
)


def test_case_trace_model_defaults() -> None:
    """验证 CaseTrace 数据模型默认字段齐全。"""
    trace = CaseTrace()
    assert trace.agent_type == ""
    assert trace.session_id == ""
    assert trace.user_id == ""
    assert isinstance(trace.messages, list)
    assert isinstance(trace.node_executions, list)
    assert isinstance(trace.tool_executions, list)
    assert isinstance(trace.plan_steps, list)
    assert trace.final_response == ""
    assert trace.memory_context == ""
    assert trace.error_message is None
    assert trace.error_traceback is None


def test_case_result_contains_trace() -> None:
    """验证 CaseResult 包含 trace 字段。"""
    case = EvalCase(id="x1", user="test", expected_tools=[], category="ambiguous")
    result = CaseResult(case=case)
    assert isinstance(result.trace, CaseTrace)


def test_react_case_records_trace_fields() -> None:
    """验证单工具(ReAct)场景 trace 中 agent_type、messages、node_executions 被填充。"""
    case = EvalCase(
        id="tr_react_1",
        user="现在车速多少",
        expected_tools=["get_vehicle_status"],
        category="vehicle_query",
    )
    result: CaseResult = Evaluator().evaluate_case(case)

    assert result.trace.agent_type == "react"
    assert result.trace.session_id.startswith("eval-react-")
    assert result.trace.user_id == "eval-user"
    assert len(result.trace.messages) > 0
    assert len(result.trace.node_executions) > 0

    node_names = [n.node_name for n in result.trace.node_executions]
    assert "retrieve_memory" in node_names
    assert "agent" in node_names
    assert "extract_memory" in node_names


def test_react_case_records_tool_executions() -> None:
    """验证 ReAct 场景 trace 中记录了工具执行。"""
    case = EvalCase(
        id="tr_react_2",
        user="现在车速多少",
        expected_tools=["get_vehicle_status"],
        category="vehicle_query",
    )
    result = Evaluator().evaluate_case(case)

    assert len(result.actual_tools) >= 1
    assert len(result.trace.tool_executions) >= 1
    assert result.trace.final_response != ""


def test_plan_case_records_plan_steps() -> None:
    """验证多工具(Plan-Execute)场景 trace 中 plan_steps 和 respond 节点被记录。"""
    case = EvalCase(
        id="tr_plan_1",
        user="去公司顺便播放音乐",
        expected_tools=["search_destination", "start_navigation", "play_media"],
        category="multi_step",
    )
    result = Evaluator().evaluate_case(case)

    assert result.trace.agent_type == "plan_execute"
    assert result.trace.session_id.startswith("eval-plan-")
    assert len(result.trace.plan_steps) >= 1

    node_names = [n.node_name for n in result.trace.node_executions]
    assert "planner" in node_names
    assert "respond" in node_names
    assert result.trace.final_response != ""


def test_plan_case_records_node_execution_latency() -> None:
    """验证 Plan 模式下每个节点记录均包含延迟信息。"""
    case = EvalCase(
        id="tr_plan_2",
        user="去公司顺便播放音乐",
        expected_tools=["search_destination", "start_navigation", "play_media"],
        category="multi_step",
    )
    result = Evaluator().evaluate_case(case)

    for node in result.trace.node_executions:
        assert isinstance(node, NodeExecutionRecord)
        assert node.latency_ms >= 0
        assert node.node_name != ""
        assert isinstance(node.input_data, dict)
        assert isinstance(node.output_data, dict)


def test_error_case_records_traceback() -> None:
    """验证失败 case 的 trace 记录完整堆栈(通过 run 异常吞掉路径)。"""

    class _BadEvaluator(Evaluator):
        def evaluate_case(self, case: EvalCase) -> CaseResult:
            raise RuntimeError("boom: simulated failure")

    case = EvalCase(id="tr_err_1", user="test", expected_tools=[], category="ambiguous")
    results = _BadEvaluator().run([case])
    assert len(results) == 1
    r = results[0]
    assert r.success is False
    assert r.trace.error_message is not None
    assert "boom" in r.trace.error_message
    assert r.trace.error_traceback is not None
    assert "RuntimeError" in r.trace.error_traceback


def test_message_record_serializable() -> None:
    """验证 MessageRecord 可通过 model_dump 序列化为 JSON 安全字典。"""
    from langchain_core.messages import HumanMessage, AIMessage

    human = HumanMessage(content="你好")
    rec1 = _message_to_record(human)
    assert rec1.role == "human"
    assert rec1.content == "你好"
    dumped = rec1.model_dump(mode="json")
    assert isinstance(dumped, dict)
    assert dumped["role"] == "human"

    ai = AIMessage(content="好的", tool_calls=[{"name": "set_temperature", "args": {"value": 24}, "id": "tc1", "type": "tool_call"}])
    rec2 = _message_to_record(ai)
    assert rec2.role == "ai"
    assert len(rec2.tool_calls) == 1
    assert rec2.tool_calls[0]["name"] == "set_temperature"


def test_run_eval_trace_write_helpers(tmp_path: Path) -> None:
    """验证 run_eval 中的 trace 写出辅助函数生成正确 JSON。"""
    from scripts.run_eval import _write_case_trace, _write_traces_index

    case = EvalCase(id="w1", user="hello", expected_tools=["get_vehicle_status"], category="vehicle_query")
    trace = CaseTrace(
        agent_type="react",
        session_id="s1",
        user_id="u1",
        messages=[MessageRecord(role="human", content="hello")],
        node_executions=[NodeExecutionRecord(node_name="agent", latency_ms=10.0)],
        tool_executions=[ToolExecutionRecord(tool_name="get_vehicle_status", result="speed=60")],
        plan_steps=[],
        final_response="speed 60km/h",
        memory_context="",
        error_message=None,
        error_traceback=None,
    )
    result = CaseResult(
        case=case,
        actual_tools=["get_vehicle_status"],
        actual_arguments=[{"dummy": True}],
        success=True,
        latency_ms=123.4,
        hallucinated_tools=[],
        trace=trace,
    )

    import sys
    import importlib.util
    scripts_dir = Path(__file__).resolve().parents[2]
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if "scripts" not in sys.modules:
        spec = importlib.util.spec_from_file_location("scripts", str(scripts_dir / "scripts" / "__init__.py"))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["scripts"] = mod

    import json
    trace_file = _write_case_trace(result)
    assert trace_file.exists()
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert data["case_id"] == "w1"
    assert data["success"] is True
    assert data["trace"]["agent_type"] == "react"
    assert data["trace"]["final_response"] == "speed 60km/h"
    assert len(data["trace"]["messages"]) == 1
    assert len(data["trace"]["node_executions"]) == 1
    assert len(data["trace"]["tool_executions"]) == 1

    index_file = _write_traces_index([result])
    assert index_file.exists()
    idx = json.loads(index_file.read_text(encoding="utf-8"))
    assert isinstance(idx, list)
    assert len(idx) == 1
    assert idx[0]["case_id"] == "w1"
    assert idx[0]["has_error"] is False


def test_sanitize_state_handles_messages() -> None:
    """验证 _sanitize_state 能安全处理含 BaseMessage 的状态字典。"""
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="hi")],
        "user_id": "u1",
        "nested": {"a": 1},
    }
    sanitized = Evaluator._sanitize_state(state)
    assert sanitized["user_id"] == "u1"
    assert isinstance(sanitized["messages"], list)
    assert sanitized["messages"][0]["role"] == "human"
    assert sanitized["messages"][0]["content"] == "hi"
