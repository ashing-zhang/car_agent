# Bad Case 修复验证单元测试
# 运行指南: pytest tests/unit/test_bad_case_fixes.py -v
# 覆盖修复:
#   - LLMPlanner._validate_navigation_integrity 导航两步走校验
#   - PlanExecuteState Reducer(plan=replace, 非追加)
#   - LLMPlanner System Prompt 强约束关键词
#   - Planner._extract_destination 对"回家/去商场"等场景正确提取

import operator
from typing import Annotated, TypedDict, get_type_hints

from langchain_core.messages import HumanMessage

from app.agent.llm_planner import LLMPlanner, _SYSTEM_PROMPT
from app.agent.plan_graph import PlanExecuteState, build_plan_agent
from app.agent.planner import Planner
from app.tools.schemas import PlanStep


def test_navigation_integrity_valid_two_step() -> None:
    """验证search→start相邻两步通过完整性校验。"""
    steps = [
        PlanStep(id=1, tool="search_destination", arguments={"query": "公司"}),
        PlanStep(id=2, tool="start_navigation", arguments={"destination": "公司 大厦"}),
    ]
    assert LLMPlanner._validate_navigation_integrity(steps) is True


def test_navigation_integrity_missing_start() -> None:
    """验证缺少start_navigation时完整性校验失败。"""
    steps = [
        PlanStep(id=1, tool="search_destination", arguments={"query": "公司"}),
        PlanStep(id=2, tool="play_media", arguments={"playlist": "默认歌单"}),
    ]
    assert LLMPlanner._validate_navigation_integrity(steps) is False


def test_navigation_integrity_no_navigation() -> None:
    """验证无导航意图时完整性校验通过。"""
    steps = [
        PlanStep(id=1, tool="play_media", arguments={"playlist": "默认歌单"}),
    ]
    assert LLMPlanner._validate_navigation_integrity(steps) is True


def test_navigation_integrity_search_not_consecutive() -> None:
    """验证search后隔一步才start也校验失败。"""
    steps = [
        PlanStep(id=1, tool="search_destination", arguments={"query": "公司"}),
        PlanStep(id=2, tool="play_media", arguments={"playlist": "默认歌单"}),
        PlanStep(id=3, tool="start_navigation", arguments={"destination": "公司 大厦"}),
    ]
    assert LLMPlanner._validate_navigation_integrity(steps) is False


def test_navigation_integrity_search_last_position() -> None:
    """验证search作为最后一步(无start)也校验失败。"""
    steps = [
        PlanStep(id=1, tool="search_destination", arguments={"query": "公司"}),
    ]
    assert LLMPlanner._validate_navigation_integrity(steps) is False


def test_plan_execute_state_plan_no_reducer() -> None:
    """验证 PlanExecuteState.plan 字段未声明Annotated[operator.add],即默认 replace。"""
    hints = get_type_hints(PlanExecuteState, include_extras=True)
    plan_hint = hints["plan"]
    assert not hasattr(plan_hint, "__metadata__") or operator.add not in [
        m for m in getattr(plan_hint, "__metadata__", [])
    ]


def test_plan_execute_state_tool_calls_has_add_reducer() -> None:
    """验证 tool_calls/tool_results/messages 字段声明了 Annotated[operator.add]。"""
    hints = get_type_hints(PlanExecuteState, include_extras=True)
    for field in ("messages", "tool_calls", "tool_results"):
        hint = hints[field]
        metadata = getattr(hint, "__metadata__", [])
        assert operator.add in metadata, f"{field} 未声明 operator.add Reducer"


def test_plan_agent_plan_not_duplicated() -> None:
    """验证 execute 循环后 plan 列表不会重复追加,即工具数不会超过步骤数。"""
    agent = build_plan_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content="我要去公司顺便播放音乐")], "user_id": "u1", "session_id": "s"}
    )
    tool_calls = result.get("tool_calls", [])
    expected_count = 3  # search_destination + start_navigation + play_media
    assert len(tool_calls) == expected_count, (
        f"期望 3 次工具调用,实际 {len(tool_calls)} 次。若>3说明plan重复追加Reducer bug未修复。"
        f" tools={[tc.name for tc in tool_calls]}"
    )
    names = [tc.name for tc in tool_calls]
    assert names.count("search_destination") == 1, "search_destination 不应被重复执行"
    assert names.count("start_navigation") == 1
    assert names.count("play_media") == 1


def test_planner_extract_home_destination() -> None:
    """验证 ms_004 场景:回家路上播放音乐 能正确提取 家 作为目的地。"""
    planner = Planner()
    dest = planner._extract_destination("回家路上播放音乐")
    assert dest is not None, "'回家路上' 中 '家' 未被识别为目的地"
    assert dest.strip() == "家"


def test_planner_extract_destination_with_stop_words() -> None:
    """验证 顺便/路上 等停止词能正确截断 ms_001/003 场景。"""
    planner = Planner()
    for text, expected in [
        ("我要去公司顺便播放音乐", "公司"),
        ("去商场顺便放点音乐", "商场"),
        ("去学校顺便放音乐", "学校"),
    ]:
        dest = planner._extract_destination(text)
        assert dest == expected, f"输入 '{text}' 应提取目的地 '{expected}',实际 '{dest}'"


def test_system_prompt_navigation_two_step_constraint() -> None:
    """验证 LLMPlanner System Prompt 包含导航两步走强约束关键词。"""
    assert "强制导航两步走" in _SYSTEM_PROMPT
    assert "search_destination" in _SYSTEM_PROMPT
    assert "start_navigation" in _SYSTEM_PROMPT
    assert "不可省略" in _SYSTEM_PROMPT


def test_system_prompt_poi_heuristic() -> None:
    """验证 LLMPlanner Prompt 包含 POI 启发式关键词,避免 ms_004 回家误判。"""
    assert "POI 启发式" in _SYSTEM_PROMPT or "POI启发式" in _SYSTEM_PROMPT
    assert '"家"' in _SYSTEM_PROMPT
    assert '"公司"' in _SYSTEM_PROMPT
    assert "回家" in _SYSTEM_PROMPT


def test_system_prompt_multi_intent_order() -> None:
    """验证 Prompt 包含多意图并列顺序约束:先导航两步再媒体。"""
    assert "涉及多种意图并列" in _SYSTEM_PROMPT
    assert "完整的导航两步" in _SYSTEM_PROMPT
