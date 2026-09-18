# LLM 意图分类器 Infrastructure 实现 - 利用 LLM 语义理解进行细粒度分类
# 运行指南:
#   from app.agent.intent.llm_classifier import LLMIntentClassifier
#   from app.llm.provider import get_llm_provider
#   cls = LLMIntentClassifier(get_llm_provider(), fallback=RuleBasedClassifier())
#   result = cls.classify("去公司顺便播放音乐")
# 若 LLM 调用失败会自动降级到 fallback 分类器

import json
import logging

from app.agent.intent.classifier import IntentClassifier
from app.agent.intent.models import AgentType, IntentClassification
from app.agent.intent.rule_classifier import RuleBasedClassifier

logger = logging.getLogger(__name__)

_CLASSIFY_SYSTEM_PROMPT = """你是车载 Agent 的意图分类器,根据用户消息判断使用单步 ReAct 模式还是多步 Plan-Execute 模式。

判断标准:
- PLAN_EXECUTE: 需要多步操作(导航+播放音乐、导航本身两步、设置温度同时开空调等)
- REACT: 单步操作(查询状态、调温、简单播放、闲聊等)

输出 JSON: {"type": "react"|"plan_execute", "confidence": 0.0~1.0, "reason": "...", "signals": {"category1": n, ...}}
只输出 JSON,不要其他内容。"""


class LLMIntentClassifier(IntentClassifier):
    """基于 LLM 的意图分类器,支持语义理解,失败自动降级到规则分类器。"""

    def __init__(
        self,
        llm: object,
        fallback: IntentClassifier | None = None,
    ) -> None:
        """注入 LLM 与 fallback 分类器。"""
        self._llm = llm
        self._fallback = fallback or RuleBasedClassifier()

    def classify(self, user_message: str, user_id: str = "demo-user") -> IntentClassification:
        """调用 LLM 进行语义分类,失败降级。"""
        try:
            return self._classify_with_llm(user_message, user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM intent classification failed for user=%s: %s, fallback to rule-based",
                user_id, exc,
            )
            return self._fallback.classify(user_message, user_id)

    def _classify_with_llm(self, user_message: str, user_id: str) -> IntentClassification:
        """调用 LLM 并解析 JSON 输出。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=f"用户消息: {user_message}"),
        ]
        response = self._llm.invoke(messages)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)

        type_str = data.get("type", "react")
        agent_type = AgentType.PLAN_EXECUTE if type_str == "plan_execute" else AgentType.REACT
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reason = str(data.get("reason", ""))
        signals = data.get("signals", {})
        if not isinstance(signals, dict):
            signals = {}

        logger.info(
            "LLM intent classified user=%s type=%s conf=%.2f",
            user_id, agent_type, confidence,
        )
        return IntentClassification(
            agent_type=agent_type,
            confidence=confidence,
            reason=reason,
            signals={k: int(v) for k, v in signals.items() if isinstance(v, (int, float))},
        )
