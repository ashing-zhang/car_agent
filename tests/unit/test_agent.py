# Agent 单元测试 - 使用 MockLLM + 场景池车辆,不依赖 API key 与外部仿真器
# 运行指南: pytest tests/unit/test_agent.py -v

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent
from app.config import PolicyConfig
from app.llm.provider import MockLLMProvider
from app.simulation.providers import SceneVehicleProvider
from app.simulation.scene_pool import get_scene_pool
from app.tools.vehicle import VehicleService


def _make_agent():
    """构造基于 Mock LLM 与场景池车辆的测试 Agent。"""
    service = VehicleService(SceneVehicleProvider(get_scene_pool()), PolicyConfig())
    llm = MockLLMProvider()
    return build_agent(service, llm)


def _tool_call_names(messages) -> list[str]:
    """提取所有 AIMessage 中的工具调用名称。"""
    names: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                names.append(tc["name"])
    return names


def test_agent_speed_query_invokes_tool() -> None:
    """验证"现在车速多少"触发 get_vehicle_status 工具。"""
    agent = _make_agent()
    result = agent.invoke({"messages": [HumanMessage(content="现在车速多少?")]})
    names = _tool_call_names(result["messages"])
    assert "get_vehicle_status" in names
    final = result["messages"][-1]
    assert "车速" in final.content


def test_agent_set_temperature_invokes_tool() -> None:
    """验证"把温度调到24度"触发 set_temperature(24)。"""
    agent = _make_agent()
    result = agent.invoke({"messages": [HumanMessage(content="把温度调到24度")]})
    names = _tool_call_names(result["messages"])
    assert "set_temperature" in names
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "set_temperature":
                    assert tc["args"]["temperature_c"] == 24.0


def test_agent_policy_violation_returns_error() -> None:
    """验证超出范围的温度(40)被 policy 拒绝并告知用户。"""
    agent = _make_agent()
    result = agent.invoke({"messages": [HumanMessage(content="把温度调到40度")]})
    final = result["messages"][-1]
    assert "超出" in final.content or "无法" in final.content


def test_agent_ac_toggle() -> None:
    """验证"开空调"触发 set_ac(True)。"""
    agent = _make_agent()
    result = agent.invoke({"messages": [HumanMessage(content="帮我开空调")]})
    names = _tool_call_names(result["messages"])
    assert "set_ac" in names


def test_agent_chatty_fallback() -> None:
    """验证无工具意图时返回文本回复。"""
    agent = _make_agent()
    result = agent.invoke({"messages": [HumanMessage(content="你好")]})
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content != ""
