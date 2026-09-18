# Agent Orchestrator 与评估集成单元测试 - 路由逻辑 + metrics 计算
# 运行指南:
#   python -m pytest tests/unit/test_orchestrator.py -v

from __future__ import annotations

from app.agent.intent.models import AgentType, IntentClassification
from app.agent.orchestrator import (
    AgentOrchestrator,
    OrchestratorResult,
    _normalize_plan_output,
    _normalize_react_output,
)
from app.config import IntentClassifierConfig
from app.evaluation.metrics import CaseResult, CaseTrace, compute_metrics
from app.evaluation.dataset import EvalCase
from app.tools.schemas import ToolCallRecord


class _StubClassifier:
    """可配置的 Stub 分类器:按消息关键字返回类型,便于隔离测试编排器。"""

    def __init__(self, forced: AgentType | None = None) -> None:
        """forced 非空时忽略消息直接返回指定类型。"""
        self.forced = forced
        self.last_message: str = ""

    def classify(self, user_message: str, user_id: str = "demo-user") -> IntentClassification:
        """返回预设类型或按关键词(导航/顺便)判定 PLAN_EXECUTE。"""
        self.last_message = user_message
        if self.forced is not None:
            t = self.forced
        elif "导航" in user_message or "顺便" in user_message or "去公司" in user_message:
            t = AgentType.PLAN_EXECUTE
        else:
            t = AgentType.REACT
        return IntentClassification(
            agent_type=t,
            confidence=0.9,
            reason="stub classifier",
            signals={"stub": 1},
        )


class _StubAgent:
    """可注入的假 Agent,invoke 返回可配置结果。"""

    def __init__(self, react: bool = True) -> None:
        """react=True 返回 ReAct 结构,否则返回 Plan-Execute 结构。"""
        self.react = react
        self.invocation_count = 0

    def invoke(self, state: dict) -> dict:
        """模拟 invoke。"""
        self.invocation_count += 1
        if self.react:
            from langchain_core.messages import AIMessage, HumanMessage

            return {
                "messages": [
                    HumanMessage(content=state.get("messages", [])[0].content),
                    AIMessage(
                        content="已调整温度",
                        tool_calls=[{"id": "call_1", "name": "set_temperature", "args": {"temperature_c": 24}}],
                    ),
                ],
            }
        return {
            "tool_calls": [
                ToolCallRecord(name="search_destination", arguments={"query": "公司"}),
                ToolCallRecord(name="start_navigation", arguments={"destination": "公司 大厦"}),
            ],
            "response": "已为您完成以下操作:\n- search_destination: OK\n- start_navigation: OK",
        }


def test_orchestrator_routes_react_for_temp_message() -> None:
    """温度调节消息应路由到 ReAct Agent,并正确归一化工具调用。"""
    orch = AgentOrchestrator(
        classifier=_StubClassifier(forced=AgentType.REACT),
        react_agent=_StubAgent(react=True),
        plan_agent=_StubAgent(react=False),
    )
    result: OrchestratorResult = orch.run("把温度调到24度", user_id="u1", session_id="s1")
    assert result["agent_type"] == AgentType.REACT
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "set_temperature"
    assert "温度" in result["response"]
    cls = result["classification"]
    assert cls is not None
    assert cls.agent_type == AgentType.REACT


def test_orchestrator_routes_plan_for_navigation() -> None:
    """导航消息应路由到 Plan-Execute Agent,并归一化 tool_calls 与 response。"""
    plan_agent = _StubAgent(react=False)
    orch = AgentOrchestrator(
        classifier=_StubClassifier(forced=AgentType.PLAN_EXECUTE),
        react_agent=_StubAgent(react=True),
        plan_agent=plan_agent,
    )
    result = orch.run("去公司顺便播放音乐", user_id="u2")
    assert result["agent_type"] == AgentType.PLAN_EXECUTE
    assert plan_agent.invocation_count == 1
    tc_names = [tc["name"] for tc in result["tool_calls"]]
    assert "search_destination" in tc_names
    assert "start_navigation" in tc_names
    assert "已为您完成" in result["response"]


def test_orchestrator_confidence_downgrade() -> None:
    """PLAN 分类置信度低于阈值时,应降级为 REACT 路由。"""
    cfg = IntentClassifierConfig(min_confidence_for_plan=0.95)
    low_conf_cls = _StubClassifier(forced=AgentType.PLAN_EXECUTE)
    low_conf_cls.classify = lambda msg, user_id="demo": IntentClassification(
        agent_type=AgentType.PLAN_EXECUTE, confidence=0.6, reason="low conf",
    )  # type: ignore[method-assign]
    react_agent = _StubAgent(react=True)
    orch = AgentOrchestrator(
        classifier=low_conf_cls,  # type: ignore[arg-type]
        react_agent=react_agent,
        plan_agent=_StubAgent(react=False),
        classifier_config=cfg,
    )
    result = orch.run("导航去公司")
    assert result["agent_type"] == AgentType.REACT
    assert react_agent.invocation_count == 1
    assert result["classification"].confidence == 0.6


def test_normalize_plan_output_extracts_tool_calls_and_response() -> None:
    """归一化 Plan-Execute 输出工具调用并拼接响应。"""
    state = {
        "tool_calls": [ToolCallRecord(name="play_media", arguments={"playlist": "x"})],
        "response": "操作完成",
    }
    resp, tcs = _normalize_plan_output(state)
    assert resp == "操作完成"
    assert tcs == [{"name": "play_media", "arguments": {"playlist": "x"}}]


def test_normalize_plan_output_handles_missing_fields() -> None:
    """缺少 tool_calls/response 时归一化不抛异常。"""
    resp, tcs = _normalize_plan_output({})
    assert resp == ""
    assert tcs == []


def test_compute_metrics_intent_accuracy_and_confusion() -> None:
    """compute_metrics 应正确计算意图分类准确率与混淆矩阵。"""
    cases = [
        EvalCase(id="n1", user="导航公司", expected_tools=["search_destination", "start_navigation"], category="navigation"),
        EvalCase(id="v1", user="调到24度", expected_tools=["set_temperature"], category="vehicle_control"),
        EvalCase(id="m1", user="去公司顺便播放音乐", expected_tools=["search_destination", "start_navigation", "play_media"], category="multi_step"),
    ]
    results: list[CaseResult] = []
    ground_truths = [AgentType.PLAN_EXECUTE, AgentType.REACT, AgentType.PLAN_EXECUTE]
    predictions = [AgentType.PLAN_EXECUTE, AgentType.REACT, AgentType.REACT]  # 第三个分类错误
    for case, gt, pred in zip(cases, ground_truths, predictions):
        trace = CaseTrace(
            intent_classified_type=str(pred),
            intent_ground_truth=str(gt),
            intent_correct=(str(pred) == str(gt)),
        )
        results.append(CaseResult(case=case, success=True, trace=trace, intent_correct=trace.intent_correct))

    report = compute_metrics(results)
    assert report.total_cases == 3
    assert abs(report.intent_classification_accuracy - 2 / 3) < 1e-6
    assert "plan_execute" in report.intent_confusion
    assert "react" in report.intent_confusion
    assert report.intent_confusion["plan_execute"]["react"] == 1  # gt=plan, pred=react 一次错误
    assert report.intent_confusion["react"]["react"] == 1
    assert report.intent_confusion["plan_execute"]["plan_execute"] == 1

    nav_cat = report.by_category["navigation"]
    assert nav_cat["intent_correct"] == 1
    assert nav_cat["intent_accuracy"] == 1.0

    ms_cat = report.by_category["multi_step"]
    assert ms_cat["intent_correct"] == 0
    assert ms_cat["intent_accuracy"] == 0.0
