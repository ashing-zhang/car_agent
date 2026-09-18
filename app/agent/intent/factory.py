# 意图分类器工厂 - 按配置实例化分类器(配置驱动,开闭原则)
# 运行指南:
#   from app.agent.intent.factory import build_intent_classifier
#   classifier = build_intent_classifier()  # 使用默认配置
#   classifier = build_intent_classifier(backend="llm")  # 强制 LLM 分类器

import logging

from app.agent.intent.classifier import IntentClassifier
from app.agent.intent.llm_classifier import LLMIntentClassifier
from app.agent.intent.rule_classifier import RuleBasedClassifier
from app.config import IntentClassifierConfig

logger = logging.getLogger(__name__)


def build_intent_classifier(
    config: IntentClassifierConfig | None = None,
    llm: object | None = None,
) -> IntentClassifier:
    """按 backend 配置构造分类器: rule | llm。"""
    if config is None:
        config = IntentClassifierConfig()

    backend = (config.backend or "rule").lower()

    if backend == "llm":
        if llm is None:
            from app.llm.provider import get_llm_provider

            llm = get_llm_provider()
        logger.info("Intent classifier backend=llm created")
        return LLMIntentClassifier(llm, fallback=RuleBasedClassifier())

    if backend != "rule":
        logger.warning("Unknown intent classifier backend=%s, fallback to rule-based", backend)

    logger.info("Intent classifier backend=rule created")
    return RuleBasedClassifier()
