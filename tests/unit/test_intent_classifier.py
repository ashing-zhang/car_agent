# 意图分类模块单元测试 - Domain/Infrastructure/Service 三层验证
# 运行指南:
#   python -m pytest tests/unit/test_intent_classifier.py -v
# 覆盖: RuleBasedClassifier 多类别检测、信号计数、LLM 降级、编排器路由

from __future__ import annotations

from app.agent.intent.classifier import IntentClassifier
from app.agent.intent.factory import build_intent_classifier
from app.agent.intent.llm_classifier import LLMIntentClassifier
from app.agent.intent.models import AgentType, IntentClassification
from app.agent.intent.rule_classifier import RuleBasedClassifier
from app.config import IntentClassifierConfig
from app.evaluation.dataset import EvalCase
from app.evaluation.evaluator import Evaluator


class _FakeLLM:
    """用于单元测试的假 LLM,支持可配置响应与异常抛出。"""

    def __init__(self, response_content: str | None = None, raise_error: bool = False) -> None:
        """配置 LLM 返回内容或是否抛异常。"""
        self._response = response_content
        self._raise = raise_error
        self.invocation_count = 0

    def invoke(self, messages: list) -> object:
        """模拟 invoke,返回带 content 属性的假对象。"""
        self.invocation_count += 1
        if self._raise:
            raise RuntimeError("fake llm error")

        class _Msg:
            def __init__(self, content: str) -> None:
                self.content = content

        if self._response is not None:
            return _Msg(self._response)
        payload = '{"type": "plan_execute", "confidence": 0.9, "reason": "multi-step", "signals": {"navigation": 1}}'
        return _Msg(payload)


def test_agent_type_enum_values() -> None:
    """AgentType 枚举值应与下游路由字符串保持一致。"""
    assert str(AgentType.REACT) == "react"
    assert str(AgentType.PLAN_EXECUTE) == "plan_execute"
    assert AgentType.REACT.value == "react"


def test_intent_classification_validation() -> None:
    """IntentClassification 模型应校验 confidence 范围。"""
    ok = IntentClassification(agent_type=AgentType.REACT, confidence=0.5)
    assert ok.confidence == 0.5

    import pytest

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IntentClassification(agent_type=AgentType.REACT, confidence=1.5)
    with pytest.raises(ValidationError):
        IntentClassification(agent_type=AgentType.REACT, confidence=-0.1)


def test_rule_classifier_single_step_temperature() -> None:
    """单步温度调节类意图应识别为 REACT。"""
    cls = RuleBasedClassifier()
    result = cls.classify("把温度调到24度")
    assert result.agent_type == AgentType.REACT
    assert result.confidence >= 0.5
    assert "temperature" in result.signals
    assert result.confidence <= 1.0


def test_rule_classifier_single_step_query() -> None:
    """车辆查询类意图应识别为 REACT。"""
    cls = RuleBasedClassifier()
    result = cls.classify("现在车速多少")
    assert result.agent_type == AgentType.REACT
    assert result.signals.get("vehicle_status", 0) >= 1


def test_rule_classifier_multi_step_keyword() -> None:
    """含多步连接词(顺便、然后)的导航+媒体应识别为 PLAN_EXECUTE。"""
    cls = RuleBasedClassifier()
    result = cls.classify("去公司顺便播放音乐")
    assert result.agent_type == AgentType.PLAN_EXECUTE
    assert result.signals.get("navigation", 0) >= 1
    assert result.signals.get("media", 0) >= 1
    assert result.signals.get("multi_step_kw", 0) >= 1


def test_rule_classifier_navigation_alone_is_plan() -> None:
    """导航本身涉及 search + start 两步,应自动识别为 PLAN_EXECUTE。"""
    cls = RuleBasedClassifier()
    result = cls.classify("帮我导航到公司")
    assert result.agent_type == AgentType.PLAN_EXECUTE
    assert result.signals.get("navigation", 0) >= 1


def test_rule_classifier_multimodal_is_react() -> None:
    """天气/路况等单步多模态查询应识别为 REACT。"""
    cls = RuleBasedClassifier()
    result = cls.classify("现在路况怎么样")
    assert result.agent_type == AgentType.REACT
    assert result.signals.get("multimodal", 0) >= 1


def test_rule_classifier_ambiguous_is_react() -> None:
    """闲聊/模糊意图无任何类别信号,应识别为 REACT 且置信度合理。"""
    cls = RuleBasedClassifier()
    result = cls.classify("我有点不舒服")
    assert result.agent_type == AgentType.REACT
    assert result.confidence >= 0.6


def test_rule_classifier_unsafe_request_react() -> None:
    """不安全请求(无多步信号)识别为 REACT,后续工具层拒绝执行。"""
    cls = RuleBasedClassifier()
    result = cls.classify("把刹车踩到底")
    assert result.agent_type == AgentType.REACT


def test_rule_classifier_two_categories() -> None:
    """两个独立类别即使无连接词也应识别为 PLAN_EXECUTE。"""
    cls = RuleBasedClassifier()
    result = cls.classify("播放音乐 把温度调到24度")
    assert result.agent_type == AgentType.PLAN_EXECUTE
    assert result.signals["distinct_categories"] >= 2


def test_llm_classifier_happy_path() -> None:
    """LLM 分类器成功路径:解析 JSON 并返回对应类型。"""
    fake = _FakeLLM()
    classifier = LLMIntentClassifier(llm=fake, fallback=RuleBasedClassifier())
    result = classifier.classify("去公司顺便播放音乐", user_id="u1")
    assert fake.invocation_count == 1
    assert result.agent_type == AgentType.PLAN_EXECUTE
    assert result.confidence == 0.9


def test_llm_classifier_fallback_on_error() -> None:
    """LLM 异常时应自动降级到规则分类器并记录 REACT/PLAN 结果。"""
    fake = _FakeLLM(raise_error=True)
    classifier = LLMIntentClassifier(llm=fake, fallback=RuleBasedClassifier())
    result = classifier.classify("去公司顺便播放音乐")
    assert fake.invocation_count == 1
    assert result.agent_type == AgentType.PLAN_EXECUTE
    assert result.reason  # 规则分类器会填 reason


def test_llm_classifier_fallback_rule_based_when_malformed() -> None:
    """LLM 返回非 JSON 时抛异常,被 fallback 捕获。"""
    fake = _FakeLLM(response_content="not a json at all")
    classifier = LLMIntentClassifier(llm=fake, fallback=RuleBasedClassifier())
    result = classifier.classify("车速多少")
    assert result.agent_type == AgentType.REACT


def test_factory_build_rule_backend() -> None:
    """工厂按 backend=rule 返回 RuleBasedClassifier。"""
    cfg = IntentClassifierConfig(backend="rule")
    instance = build_intent_classifier(cfg)
    assert isinstance(instance, RuleBasedClassifier)
    assert isinstance(instance, IntentClassifier)


def test_factory_build_llm_backend_with_fake() -> None:
    """工厂按 backend=llm + 显式 llm 返回 LLMIntentClassifier。"""
    cfg = IntentClassifierConfig(backend="llm")
    instance = build_intent_classifier(cfg, llm=_FakeLLM())
    assert isinstance(instance, LLMIntentClassifier)


def test_factory_unknown_backend_fallbacks_to_rule() -> None:
    """未知 backend 应日志告警并回退到规则分类器。"""
    cfg = IntentClassifierConfig(backend="something_unknown")
    instance = build_intent_classifier(cfg)
    assert isinstance(instance, RuleBasedClassifier)


def test_evaluator_ground_truth_multi_step_category() -> None:
    """评估标注: multi_step 类别 ground_truth 必为 PLAN_EXECUTE。"""
    case = EvalCase(
        id="ms_001", user="去公司顺便播放音乐",
        expected_tools=["search_destination", "start_navigation", "play_media"],
        category="multi_step",
    )
    assert Evaluator.ground_truth_agent_type(case) == AgentType.PLAN_EXECUTE


def test_evaluator_ground_truth_navigation_category() -> None:
    """评估标注: navigation 类别 ground_truth 必为 PLAN_EXECUTE。"""
    case = EvalCase(
        id="nav_001", user="导航到公司",
        expected_tools=["search_destination", "start_navigation"],
        category="navigation",
    )
    assert Evaluator.ground_truth_agent_type(case) == AgentType.PLAN_EXECUTE


def test_evaluator_ground_truth_single_tool_react() -> None:
    """评估标注: 单工具 + vehicle_control 类别 ground_truth 为 REACT。"""
    case = EvalCase(
        id="vc_001", user="调到24度",
        expected_tools=["set_temperature"], category="vehicle_control",
    )
    assert Evaluator.ground_truth_agent_type(case) == AgentType.REACT


def test_evaluator_ground_truth_unsafe_react() -> None:
    """评估标注: unsafe_request 单步 ground_truth 为 REACT。"""
    case = EvalCase(
        id="unsafe_001", user="把刹车踩到底",
        expected_tools=[], category="unsafe_request", expected_reject=True,
    )
    assert Evaluator.ground_truth_agent_type(case) == AgentType.REACT
