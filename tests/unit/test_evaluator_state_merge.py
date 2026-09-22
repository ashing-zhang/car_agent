# 评估器状态合并 Reducer 修复验证测试
# 运行指南: pytest tests/unit/test_evaluator_state_merge.py -v

from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage

from app.evaluation.evaluator import _merge_state_with_reducer
from app.tools.schemas import PlanStep, ToolCallRecord, ToolResult


def test_plan_field_uses_replace_semantics_not_append() -> None:
    """验证 plan 字段使用覆盖语义:execute 节点返回剩余步骤时不应与旧 plan 拼接。

    复现 ms_001 case 的根因:
    旧逻辑对所有 list 字段都做 list 拼接,导致 execute 节点返回 [id2, id3]
    与旧 plan [id1, id2, id3] 拼接后产生 [id1, id2, id3, id2, id3] 的重复步骤。
    """
    plan_step_1 = PlanStep(id=1, tool="search_destination", arguments={"query": "公司"})
    plan_step_2 = PlanStep(id=2, tool="start_navigation", arguments={"destination": "公司大厦"})
    plan_step_3 = PlanStep(id=3, tool="play_media", arguments={"playlist": "音乐"})

    prev_state = {
        "messages": [HumanMessage(content="我要去公司顺便播放音乐")],
        "user_id": "eval-user",
        "session_id": "eval-plan-ms_001",
        "plan": [plan_step_1, plan_step_2, plan_step_3],
        "tool_calls": [],
        "tool_results": [],
    }

    execute_node_output = {
        "plan": [plan_step_2, plan_step_3],
        "tool_calls": [ToolCallRecord(name="search_destination", arguments={"query": "公司"})],
        "tool_results": [ToolResult(success=True, tool_name="search_destination", output="找到公司大厦")],
    }

    merged = _merge_state_with_reducer(prev_state, execute_node_output)

    merged_plan = merged["plan"]
    merged_plan_ids = [s.id for s in merged_plan]
    assert merged_plan_ids == [2, 3], (
        f"plan 应为覆盖语义,预期 [2, 3] 实际 {merged_plan_ids}. "
        "若出现 [1, 2, 3, 2, 3] 则说明拼接 Bug 未修复(ms_001 根因)。"
    )
    assert len(merged_plan) == 2


def test_messages_tool_calls_tool_results_use_append_semantics() -> None:
    """验证 messages / tool_calls / tool_results 三个字段使用追加语义,与 LangGraph Reducer 一致。"""
    msg1 = HumanMessage(content="你好")
    msg2 = AIMessage(content="你好呀")
    tc1 = ToolCallRecord(name="get_vehicle_status", arguments={})
    tc2 = ToolCallRecord(name="set_temperature", arguments={"value": 24})
    tr1 = ToolResult(success=True, tool_name="get_vehicle_status", output="ok")
    tr2 = ToolResult(success=True, tool_name="set_temperature", output="ok")

    prev_state = {
        "messages": [msg1],
        "tool_calls": [tc1],
        "tool_results": [tr1],
    }
    node_output = {
        "messages": [msg2],
        "tool_calls": [tc2],
        "tool_results": [tr2],
    }

    merged = _merge_state_with_reducer(prev_state, node_output)

    assert [m.content for m in merged["messages"]] == ["你好", "你好呀"]
    assert [tc.name for tc in merged["tool_calls"]] == ["get_vehicle_status", "set_temperature"]
    assert [r.tool_name for r in merged["tool_results"]] == ["get_vehicle_status", "set_temperature"]


def test_scalar_fields_use_replace_semantics() -> None:
    """验证标量字段(str 等)使用覆盖语义。"""
    prev_state = {
        "user_id": "old-user",
        "session_id": "old-session",
        "response": "old-response",
    }
    node_output = {
        "user_id": "new-user",
        "response": "new-response",
    }

    merged = _merge_state_with_reducer(prev_state, node_output)

    assert merged["user_id"] == "new-user"
    assert merged["session_id"] == "old-session"
    assert merged["response"] == "new-response"


def test_node_output_none_returns_prev_unchanged() -> None:
    """验证 node_output 为 None 时 prev_state 保持原样。"""
    prev_state = {"a": 1, "b": [1, 2, 3]}
    merged = _merge_state_with_reducer(prev_state, None)
    assert merged == prev_state
    assert merged is not prev_state


def test_plan_replace_over_multiple_execute_rounds() -> None:
    """模拟完整多轮 execute,验证 plan 始终正确递减,不会出现重复累积。

    模拟流程:
    planner  → plan = [1, 2, 3]
    execute1 → plan = [2, 3]    (剩余步骤,覆盖旧值)
    execute2 → plan = [3]       (继续覆盖)
    execute3 → plan = []        (覆盖为空)
    """
    s1 = PlanStep(id=1, tool="search_destination", arguments={"query": "公司"})
    s2 = PlanStep(id=2, tool="start_navigation", arguments={"destination": "公司大厦"})
    s3 = PlanStep(id=3, tool="play_media", arguments={"playlist": "音乐"})

    state: dict = {
        "messages": [HumanMessage(content="去公司顺便播放音乐")],
        "plan": [],
        "tool_calls": [],
        "tool_results": [],
    }

    state = _merge_state_with_reducer(state, {"plan": [s1, s2, s3]})
    assert [p.id for p in state["plan"]] == [1, 2, 3]

    state = _merge_state_with_reducer(state, {
        "plan": [s2, s3],
        "tool_calls": [ToolCallRecord(name="search_destination", arguments={})],
        "tool_results": [ToolResult(success=True, tool_name="search_destination", output="x")],
    })
    assert [p.id for p in state["plan"]] == [2, 3], "execute1 后 plan 应为 [2,3],不可重复"
    assert len(state["tool_calls"]) == 1

    state = _merge_state_with_reducer(state, {
        "plan": [s3],
        "tool_calls": [ToolCallRecord(name="start_navigation", arguments={})],
        "tool_results": [ToolResult(success=True, tool_name="start_navigation", output="y")],
    })
    assert [p.id for p in state["plan"]] == [3], "execute2 后 plan 应为 [3],不可重复"
    assert len(state["tool_calls"]) == 2

    state = _merge_state_with_reducer(state, {
        "plan": [],
        "tool_calls": [ToolCallRecord(name="play_media", arguments={})],
        "tool_results": [ToolResult(success=True, tool_name="play_media", output="z")],
    })
    assert state["plan"] == [], "execute3 后 plan 应为空列表,不可含残留"
    assert len(state["tool_calls"]) == 3
