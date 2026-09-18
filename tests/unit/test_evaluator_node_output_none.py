# 评估器 node_output None 防御修复验证测试
# 运行指南: pytest tests/unit/test_evaluator_node_output_none.py -v

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from app.evaluation.dataset import EvalCase
from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics import (
    CaseResult,
    CaseTrace,
    MessageRecord,
    NodeExecutionRecord,
    ToolExecutionRecord,
)


class _FakeMemoryService:
    """无用户偏好的 MemoryService 桩,retrieve_memory 将返回 {}。"""

    def recall_preference(self, user_id: str, predicate: str) -> None:
        """永远返回 None,模拟新用户无任何偏好。"""
        return None

    def remember(self, text: str, user_id: str, session_id: str) -> None:
        """无操作。"""
        return None


def _make_stream_yielding_none(step_outputs: list[dict[str, Any]]) -> Any:
    """构造一个 stream() 生成器,模拟 LangGraph 输出 node_output 为 None 的场景。"""

    class _FakeAgent:
        def stream(self, input_state: dict):
            """按顺序 yield 预定义的 step_outputs。"""
            for step in step_outputs:
                yield step

    return _FakeAgent()


def test_react_retrieve_memory_node_output_none_no_error() -> None:
    """验证 ReAct 模式下 retrieve_memory 返回 None 时 status 不为 error。"""
    case = EvalCase(
        id="t_nonemem_1",
        user="有点冷",
        expected_tools=["set_temperature"],
        expected_reject=False,
        category="memory",
    )

    evaluator = Evaluator()
    session_id = f"eval-react-{case.id}"
    user_id = "eval-user"
    trace = CaseTrace()
    trace.agent_type = "react"
    trace.session_id = session_id
    trace.user_id = user_id

    input_state = {
        "messages": [HumanMessage(content=case.user)],
        "user_id": user_id,
        "session_id": session_id,
    }
    prev_state: dict = dict(input_state)
    memory_ctx_parts: list[str] = []

    fake_agent = _make_stream_yielding_none([
        {"retrieve_memory": None},
    ])

    actual_tools: list[str] = []
    actual_args: list[dict] = []

    for step_output in fake_agent.stream(input_state):
        for node_name, node_output in step_output.items():
            import time as _time
            node_start = _time.perf_counter()
            node_record = NodeExecutionRecord(
                node_name=node_name,
                input_data=evaluator._sanitize_state(prev_state),
            )
            try:
                node_record.output_data = evaluator._sanitize_state(node_output)

                if node_name == "retrieve_memory":
                    sys_msgs = (node_output or {}).get("messages", [])
                    for sm in sys_msgs:
                        content = sm.content if isinstance(sm.content, str) else str(sm.content)
                        memory_ctx_parts.append(content)
                    trace.memory_context = "\n---\n".join(memory_ctx_parts)

            except Exception as exc:
                node_record.status = "error"
                node_record.error = str(exc)
            finally:
                node_record.latency_ms = (_time.perf_counter() - node_start) * 1000
                trace.node_executions.append(node_record)

            merged = dict(prev_state)
            for k, v in (node_output or {}).items():
                if isinstance(v, list) and isinstance(merged.get(k), list):
                    merged[k] = list(merged[k]) + list(v)
                else:
                    merged[k] = v
            prev_state = merged

    assert len(trace.node_executions) == 1
    node = trace.node_executions[0]
    assert node.node_name == "retrieve_memory"
    assert node.status != "error", f"retrieve_memory 返回 None 不应标记为 error,实际 error={node.error}"
    assert node.error is None
    assert node.output_data == {}
    assert trace.memory_context == ""
    assert actual_tools == []
    assert actual_args == []


def test_react_all_node_types_handle_none_safely() -> None:
    """验证 ReAct 全部节点类型(retrieve_memory/agent/tools/extract_memory)处理 None 不抛错。"""
    from langchain_core.messages import AIMessage as _AIMsg, ToolMessage as _ToolMsg

    case = EvalCase(
        id="t_nonemem_all",
        user="现在车速",
        expected_tools=["get_vehicle_status"],
        category="vehicle_query",
    )
    evaluator = Evaluator()
    session_id = f"eval-react-{case.id}"
    user_id = "eval-user"
    trace = CaseTrace()
    trace.agent_type = "react"
    trace.session_id = session_id
    trace.user_id = user_id

    input_state = {
        "messages": [HumanMessage(content=case.user)],
        "user_id": user_id,
        "session_id": session_id,
    }
    prev_state: dict = dict(input_state)
    memory_ctx_parts: list[str] = []
    actual_tools: list[str] = []
    actual_args: list[dict] = []

    ai_msg = _AIMsg(
        content="查看车速中...",
        tool_calls=[{"name": "get_vehicle_status", "args": {}, "id": "tc1", "type": "tool_call"}],
    )
    tool_msg = _ToolMsg(content="车速 60km/h", tool_call_id="tc1", name="get_vehicle_status")

    fake_agent = _make_stream_yielding_none([
        {"retrieve_memory": None},
        {"agent": {"messages": [ai_msg]}},
        {"tools": None},
        {"agent": None},
        {"extract_memory": {}},
    ])

    import time as _time
    for step_output in fake_agent.stream(input_state):
        for node_name, node_output in step_output.items():
            node_start = _time.perf_counter()
            node_record = NodeExecutionRecord(
                node_name=node_name,
                input_data=evaluator._sanitize_state(prev_state),
            )
            try:
                node_record.output_data = evaluator._sanitize_state(node_output)

                if node_name == "retrieve_memory":
                    sys_msgs = (node_output or {}).get("messages", [])
                    for sm in sys_msgs:
                        content = sm.content if isinstance(sm.content, str) else str(sm.content)
                        memory_ctx_parts.append(content)
                    trace.memory_context = "\n---\n".join(memory_ctx_parts)

                if node_name == "agent":
                    out_msgs = (node_output or {}).get("messages", [])
                    for m in out_msgs:
                        if isinstance(m, _AIMsg) and m.tool_calls:
                            for tc in m.tool_calls:
                                actual_tools.append(tc.get("name", ""))
                                args_dict = tc.get("args", {})
                                actual_args.append(dict(args_dict) if isinstance(args_dict, dict) else {})

                if node_name == "tools":
                    out_msgs = (node_output or {}).get("messages", [])
                    for m in out_msgs:
                        trec = ToolExecutionRecord(
                            tool_name="",
                            result=m.content if isinstance(m.content, str) else str(m.content),
                            success=True,
                        )
                        if isinstance(m, _ToolMsg):
                            trec.tool_name = getattr(m, "name", "") or f"tool_{m.tool_call_id}"
                        trace.tool_executions.append(trec)

            except Exception as exc:
                node_record.status = "error"
                node_record.error = str(exc)
            finally:
                node_record.latency_ms = (_time.perf_counter() - node_start) * 1000
                trace.node_executions.append(node_record)

            merged = dict(prev_state)
            for k, v in (node_output or {}).items():
                if isinstance(v, list) and isinstance(merged.get(k), list):
                    merged[k] = list(merged[k]) + list(v)
                else:
                    merged[k] = v
            prev_state = merged

    error_nodes = [n for n in trace.node_executions if n.status == "error"]
    assert len(error_nodes) == 0, f"不应有节点标记为 error,但发现: {[(n.node_name, n.error) for n in error_nodes]}"
    assert actual_tools == ["get_vehicle_status"]
    assert actual_args == [{}]


def test_plan_execute_all_node_types_handle_none_safely() -> None:
    """验证 Plan-Execute 模式所有节点(planner/execute/respond)处理 None 不抛错。"""
    case = EvalCase(
        id="t_noneplan_1",
        user="去公司播放音乐",
        expected_tools=["search_destination", "start_navigation", "play_media"],
        category="multi_step",
    )
    evaluator = Evaluator()
    session_id = f"eval-plan-{case.id}"
    user_id = "eval-user"
    trace = CaseTrace()
    trace.agent_type = "plan_execute"
    trace.session_id = session_id
    trace.user_id = user_id

    input_state: dict = {
        "messages": [HumanMessage(content=case.user)],
        "user_id": user_id,
        "session_id": session_id,
    }
    prev_state: dict = dict(input_state)
    actual_tools: list[str] = []
    actual_args: list[dict] = []

    fake_plan_agent = _make_stream_yielding_none([
        {"planner": None},
        {"execute": None},
        {"respond": None},
    ])

    import time as _time
    for step_output in fake_plan_agent.stream(input_state):
        for node_name, node_output in step_output.items():
            node_start = _time.perf_counter()
            node_record = NodeExecutionRecord(
                node_name=node_name,
                input_data=evaluator._sanitize_state(prev_state),
            )
            try:
                node_record.output_data = evaluator._sanitize_state(node_output)

                if node_name == "planner":
                    plan_steps = (node_output or {}).get("plan", [])
                    trace.plan_steps = [
                        s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in plan_steps
                    ]

                if node_name == "execute":
                    calls = (node_output or {}).get("tool_calls", [])
                    results = (node_output or {}).get("tool_results", [])
                    for tc in calls:
                        name = tc.name if hasattr(tc, "name") else tc.get("name", "")
                        args = tc.arguments if hasattr(tc, "arguments") else tc.get("arguments", {})
                        actual_tools.append(name)
                        actual_args.append(dict(args))
                    for tr in results:
                        tool_rec = ToolExecutionRecord(
                            tool_name=tr.tool_name if hasattr(tr, "tool_name") else tr.get("tool_name", ""),
                            arguments=dict(args) if args else {},
                            result=tr.output if hasattr(tr, "output") else tr.get("output", ""),
                            success=tr.success if hasattr(tr, "success") else tr.get("success", True),
                            error=tr.error if hasattr(tr, "error") else tr.get("error"),
                        )
                        trace.tool_executions.append(tool_rec)

                if node_name == "respond":
                    trace.final_response = (node_output or {}).get("response", "")

            except Exception as exc:
                node_record.status = "error"
                node_record.error = str(exc)
            finally:
                node_record.latency_ms = (_time.perf_counter() - node_start) * 1000
                trace.node_executions.append(node_record)

            merged = dict(prev_state)
            for k, v in (node_output or {}).items():
                if isinstance(v, list) and isinstance(merged.get(k), list):
                    merged[k] = list(merged[k]) + list(v)
                else:
                    merged[k] = v
            prev_state = merged

    error_nodes = [n for n in trace.node_executions if n.status == "error"]
    assert len(error_nodes) == 0, f"不应有节点标记为 error,但发现: {[(n.node_name, n.error) for n in error_nodes]}"
    assert trace.plan_steps == []
    assert trace.final_response == ""
    assert actual_tools == []
    assert actual_args == []


def test_sanitize_state_handles_none() -> None:
    """验证 _sanitize_state 传入 None 返回 {}。"""
    result = Evaluator._sanitize_state(None)
    assert result == {}
    assert isinstance(result, dict)
