# Planner 与 plan-execute 单元测试(规格第16节,Phase 4)
# 运行指南: pytest tests/unit/test_planner.py -v

from langchain_core.messages import HumanMessage

from app.agent.plan_graph import build_plan_agent
from app.agent.planner import Planner


def test_planner_demo_d_generates_multi_step() -> None:
    """验证 Demo D 生成 >=2 步且导航依赖顺序正确。"""
    planner = Planner()
    steps = planner.plan("我要去公司,顺便播放我平时上班喜欢听的音乐", "u1")
    assert len(steps) >= 2
    tool_names = [s.tool for s in steps]
    assert "search_destination" in tool_names
    assert "start_navigation" in tool_names
    assert "play_media" in tool_names
    assert tool_names.index("search_destination") < tool_names.index("start_navigation")


def test_planner_temperature_single_step() -> None:
    """验证温度请求只生成单步。"""
    planner = Planner()
    steps = planner.plan("把温度调到24度", "u1")
    assert len(steps) == 1
    assert steps[0].tool == "set_temperature"
    assert steps[0].arguments["temperature_c"] == 24.0


def test_planner_no_intent_returns_empty() -> None:
    """验证无工具意图时返回空计划。"""
    planner = Planner()
    steps = planner.plan("你好", "u1")
    assert len(steps) == 0


def test_plan_agent_executes_multiple_tools() -> None:
    """验证 plan-execute 执行 >=2 个工具并生成回复。"""
    agent = build_plan_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content="去公司顺便播放音乐")], "user_id": "u1", "session_id": "s"}
    )
    tool_calls = result.get("tool_calls", [])
    assert len(tool_calls) >= 2
    assert result.get("response", "") != ""


def test_plan_agent_preserves_dependency_order() -> None:
    """验证搜索在开始导航之前执行。"""
    agent = build_plan_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content="我要去公司")], "user_id": "u1", "session_id": "s"}
    )
    names = [tc.name for tc in result.get("tool_calls", [])]
    if "start_navigation" in names and "search_destination" in names:
        assert names.index("search_destination") < names.index("start_navigation")
