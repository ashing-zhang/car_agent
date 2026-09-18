# 规则意图分类器 Infrastructure 实现 - 基于关键词与启发式规则(配置驱动)
# 运行指南:
#   from app.agent.intent.rule_classifier import RuleBasedClassifier
#   cls = RuleBasedClassifier()
#   result = cls.classify("去公司顺便播放音乐")
# 分类规则: 检测是否存在多个意图或显式多步指示词

import logging
import re

from app.agent.intent.classifier import IntentClassifier
from app.agent.intent.models import AgentType, IntentClassification

logger = logging.getLogger(__name__)

MULTI_STEP_KEYWORDS: tuple[str, ...] = (
    "顺便", "然后", "同时", "路上", "途中", "的路上", "的时候",
    "并", "并且", "一边", "另外", "还有", "再", "还要",
    "先", "再", "之后", "接着", "随后",
)

NAVIGATION_KEYWORDS: tuple[str, ...] = (
    "导航", "去", "回", "前往", "出发去", "怎么去", "开到", "驶向",
)

NAVIGATION_PRECISE_PATTERNS: tuple[str, ...] = (
    "导航到", "导航去", "导航回", "导航前往", "帮我导航", "开始导航",
    "开去", "开车去", "开回", "驾车去", "到.*去", "到.*(公司|家|医院|机场|学校|商场|加油站|餐厅)",
)

MEDIA_KEYWORDS: tuple[str, ...] = (
    "音乐", "播放", "歌单", "听", "放点", "放首", "歌", "歌曲",
)

TEMPERATURE_KEYWORDS: tuple[str, ...] = (
    "温度", "冷", "热", "空调", "暖风", "冷气", "制冷", "制热",
)

VOLUME_KEYWORDS: tuple[str, ...] = (
    "音量", "大声", "小声", "声音",
)

VEHICLE_STATUS_KEYWORDS: tuple[str, ...] = (
    "车速", "电量", "续航", "状态", "还剩", "多少", "当前",
)

MULTIMODAL_KEYWORDS: tuple[str, ...] = (
    "路况", "天气", "交通", "周围", "看看", "下雨", "环境",
)

CATEGORY_KEYWORD_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("navigation", NAVIGATION_KEYWORDS),
    ("media", MEDIA_KEYWORDS),
    ("temperature", TEMPERATURE_KEYWORDS),
    ("volume", VOLUME_KEYWORDS),
    ("vehicle_status", VEHICLE_STATUS_KEYWORDS),
    ("multimodal", MULTIMODAL_KEYWORDS),
]


def _count_matches(text: str, keywords: tuple[str, ...]) -> int:
    """统计关键词命中次数(按不重叠出现次数计)。"""
    count = 0
    for kw in keywords:
        count += len(re.findall(re.escape(kw), text))
    return count


class RuleBasedClassifier(IntentClassifier):
    """基于关键词与启发式规则的意图分类器,零外部依赖即可运行。"""

    def classify(self, user_message: str, user_id: str = "demo-user") -> IntentClassification:
        """检测用户消息中的类别信号与多步连接词。"""
        signals: dict[str, int] = {}
        for label, keywords in CATEGORY_KEYWORD_GROUPS:
            n = _count_matches(user_message, keywords)
            if n:
                signals[label] = n

        multi_step_signals = _count_matches(user_message, MULTI_STEP_KEYWORDS)
        distinct_categories = len(signals)

        signals["multi_step_kw"] = multi_step_signals
        signals["distinct_categories"] = distinct_categories

        need_plan: bool = False
        if distinct_categories >= 2:
            need_plan = True
        elif multi_step_signals >= 1 and distinct_categories >= 1:
            need_plan = True
        elif "navigation" in signals and distinct_categories >= 1:
            need_plan = True

        if need_plan:
            base_confidence = min(0.55 + 0.1 * distinct_categories + 0.05 * multi_step_signals, 0.98)
            agent_type = AgentType.PLAN_EXECUTE
            reason_parts = [f"命中{distinct_categories}个类别"]
            if multi_step_signals:
                reason_parts.append(f"{multi_step_signals}个多步连接词")
            reason = ",".join(reason_parts)
        else:
            base_confidence = min(0.6 + 0.08 * max(signals.values(), default=0), 0.95)
            agent_type = AgentType.REACT
            if distinct_categories == 0:
                base_confidence = max(base_confidence, 0.75)
            reason = "单步或闲聊类意图" if distinct_categories == 0 else "单类别单步意图"

        logger.info(
            "Intent classified user=%s type=%s conf=%.2f signals=%s",
            user_id, agent_type, base_confidence, signals,
        )
        return IntentClassification(
            agent_type=agent_type,
            confidence=base_confidence,
            reason=reason,
            signals=signals,
        )
